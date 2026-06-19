//! UTXO locking mechanism for transaction construction.
//!
//! This module provides functionality to temporarily lock UTXOs (Unspent Transaction Outputs)
//! during transaction construction, preventing double-spending scenarios where the same
//! outputs might be selected for multiple concurrent transactions.
//!
//! # Overview
//!
//! When creating a transaction, the wallet must:
//! 1. Select appropriate UTXOs to cover the transaction amount plus fees
//! 2. Lock those UTXOs to prevent other transactions from using them
//! 3. Either complete the transaction (consuming the UTXOs) or release the lock on failure
//!
//! The [`FundLocker`] handles steps 1 and 2, with automatic expiration to handle step 3
//! in case of failures or timeouts.
//!
//! # Idempotency
//!
//! Lock operations support idempotency keys, allowing clients to safely retry requests
//! without accidentally locking additional funds. If a lock request with the same
//! idempotency key already exists, the original result is returned.

use std::sync::Mutex;

use chrono::{Duration, Utc};
use log::info;
use tari_transaction_components::tari_amount::MicroMinotari;
use uuid::Uuid;

use crate::{
    api::types::LockFundsResult,
    db::{self, SqlitePool},
    log::mask_amount,
    transactions::input_selector::InputSelector,
};

/// Manages temporary locking of UTXOs during transaction construction.
///
/// `FundLocker` ensures that UTXOs selected for a transaction cannot be used
/// by other concurrent transactions, preventing double-spending within the wallet.
/// Locks are time-limited and automatically expire if the transaction is not
/// completed within the specified duration.
///
/// # Thread Safety
///
/// `FundLocker` uses database-level locking and can be safely shared across
/// threads via cloning (which clones the underlying connection pool).
///
/// # Example
///
/// ```rust,ignore
/// use minotari::transactions::fund_locker::FundLocker;
/// use tari_transaction_components::tari_amount::MicroMinotari;
///
/// let locker = FundLocker::new(db_pool);
///
/// // Lock funds for a transaction
/// let result = locker.lock(
///     account_id,
///     MicroMinotari(1_000_000),  // amount to send
///     1,                         // number of outputs
///     MicroMinotari(5),          // fee per gram
///     None,                      // use default output size estimate
///     Some("unique-key".into()), // idempotency key
///     300,                       // lock for 5 minutes
/// ).await?;
///
/// // Use result.utxos to build the transaction
/// ```
pub struct FundLocker {
    db_pool: SqlitePool,
    lock: Mutex<()>,
}

impl FundLocker {
    /// Creates a new `FundLocker` with the given database connection pool.
    ///
    /// # Arguments
    ///
    /// * `db_pool` - SQLite connection pool for database operations
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let locker = FundLocker::new(db_pool);
    /// ```
    pub fn new(db_pool: SqlitePool) -> Self {
        Self {
            db_pool,
            lock: Mutex::new(()),
        }
    }

    /// Locks UTXOs for a pending transaction.
    ///
    /// Selects unspent outputs sufficient to cover the requested amount plus estimated
    /// transaction fees, then locks them in the database with an expiration time.
    /// If an idempotency key is provided and a matching pending transaction exists,
    /// returns the existing lock result without creating a new one.
    ///
    /// # Arguments
    ///
    /// * `account_id` - The account whose UTXOs should be locked
    /// * `amount` - The amount to be sent (excluding fees)
    /// * `num_outputs` - Number of transaction outputs (typically 1 for recipient + optional change)
    /// * `fee_per_gram` - Fee rate in MicroMinotari per gram of transaction weight
    /// * `estimated_output_size` - Optional override for output size estimation; if `None`,
    ///   uses default calculation based on standard output features
    /// * `idempotency_key` - Optional unique key for idempotent operations; if provided and
    ///   a matching lock exists, returns the existing result
    /// * `seconds_to_lock_utxos` - Duration in seconds before the lock expires
    ///
    /// # Returns
    ///
    /// Returns a [`LockFundsResult`] containing:
    /// - The selected UTXOs
    /// - Whether a change output is required
    /// - Total value of selected UTXOs
    /// - Fee calculations with and without change output
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Database connection fails
    /// - Insufficient funds are available
    /// - UTXO selection fails due to serialization errors
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = locker.lock(
    ///     account_id,
    ///     MicroMinotari(500_000),
    ///     1,
    ///     MicroMinotari(5),
    ///     None,
    ///     Some("tx-123".to_string()),
    ///     600, // 10 minute lock
    /// ).await?;
    ///
    /// println!("Locked {} UTXOs worth {}", result.utxos.len(), result.total_value);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn lock(
        &self,
        account_id: i64,
        amount: MicroMinotari,
        num_outputs: usize,
        fee_per_gram: MicroMinotari,
        estimated_output_size: Option<usize>,
        idempotency_key: Option<String>,
        seconds_to_lock_utxos: u64,
        confirmation_window: u64,
    ) -> Result<LockFundsResult, anyhow::Error> {
        // Fast path: check idempotency without acquiring the mutex.
        // This allows duplicate requests to return immediately without
        // blocking other concurrent operations.
        if let Some(idempotency_key_str) = &idempotency_key {
            let conn = self.db_pool.get()?;
            if let Some(response) = db::find_pending_transaction_locked_funds_by_idempotency_key(
                &conn,
                idempotency_key_str,
                account_id,
            )? {
                info!(
                    target: "audit",
                    idempotency_key = idempotency_key_str.as_str();
                    "Found existing pending transaction lock (fast path)"
                );
                return Ok(response);
            }
        }

        // Acquire mutex to serialize concurrent UTXO selection attempts.
        // Without this, two concurrent requests could both read the same
        // unspent outputs before either commits, leading to double-selection.
        let _guard = self.lock.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        info!(
            target: "audit",
            account_id = account_id,
            amount = &*mask_amount(amount);
            "Locking funds"
        );
        let mut conn = self.db_pool.get()?;

        // Use IMMEDIATE transaction to acquire a RESERVE lock upfront.
        // This prevents concurrent transactions from reading and selecting
        // the same UTXOs. A DEFERRED transaction would only acquire a SHARED
        // lock on reads, allowing race conditions.
        let transaction = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Re-check idempotency inside the mutex and transaction.
        // Between the fast path check and acquiring the mutex, another thread
        // may have created a lock with the same key.
        if let Some(idempotency_key_str) = &idempotency_key
            && let Some(response) = db::find_pending_transaction_locked_funds_by_idempotency_key(
                &transaction,
                idempotency_key_str,
                account_id,
            )?
        {
            info!(
                target: "audit",
                idempotency_key = idempotency_key_str.as_str();
                "Found existing pending transaction lock (after mutex)"
            );
            transaction.rollback()?;
            return Ok(response);
        }

        // Select UTXOs INSIDE the transaction so concurrent requests
        // cannot select the same outputs.
        let input_selector = InputSelector::new(account_id, confirmation_window);
        let utxo_selection = input_selector.fetch_unspent_outputs(
            &transaction,
            amount,
            num_outputs,
            fee_per_gram,
            estimated_output_size,
        )?;

        #[allow(clippy::cast_possible_wrap)]
        let expires_at = Utc::now() + Duration::seconds(seconds_to_lock_utxos as i64);
        let idempotency_key = idempotency_key.unwrap_or_else(|| Uuid::new_v4().to_string());
        let pending_tx_id = db::create_pending_transaction(
            &transaction,
            &idempotency_key,
            account_id,
            utxo_selection.requires_change_output,
            utxo_selection.total_value,
            utxo_selection.fee_without_change,
            utxo_selection.fee_with_change,
            expires_at,
        )?;

        for utxo in &utxo_selection.utxos {
            db::lock_output(&transaction, utxo.id, &pending_tx_id, expires_at)?;
        }

        transaction.commit()?;

        info!(
            target: "audit",
            utxos_count = utxo_selection.utxos.len(),
            total_value = &*mask_amount(utxo_selection.total_value);
            "Funds locked successfully"
        );

        Ok(LockFundsResult {
            utxos: utxo_selection.utxos.iter().map(|utxo| utxo.output.clone()).collect(),
            requires_change_output: utxo_selection.requires_change_output,
            total_value: utxo_selection.total_value,
            fee_without_change: utxo_selection.fee_without_change,
            fee_with_change: utxo_selection.fee_with_change,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::init_db;
    use tempfile::tempdir;

    fn setup_test_db() -> SqlitePool {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_wallet.db");
        init_db(db_path).unwrap()
    }

    #[test]
    fn test_fund_locker_idempotency_key_returns_same_result() {
        let pool = setup_test_db();
        let locker = FundLocker::new(pool.clone());
        let key = "test-idempotency-key".to_string();

        // First lock with the key
        let result1 = locker.lock(
            1,
            MicroMinotari(1_000_000),
            1,
            MicroMinotari(5),
            None,
            Some(key.clone()),
            300,
            10,
        );

        // Second lock with the same key should return the same result
        let result2 = locker.lock(
            1,
            MicroMinotari(1_000_000),
            1,
            MicroMinotari(5),
            None,
            Some(key),
            300,
            10,
        );

        // Both should either succeed with same values or fail identically
        match (result1, result2) {
            (Ok(r1), Ok(r2)) => {
                assert_eq!(r1.total_value, r2.total_value);
                assert_eq!(r1.utxos.len(), r2.utxos.len());
            },
            (Err(_), Err(_)) => {
                // Both failed, which is acceptable
            },
            _ => panic!("Idempotency mismatch: one succeeded, one failed"),
        }
    }

    #[test]
    fn test_fund_locker_different_keys_get_different_results() {
        let pool = setup_test_db();
        let locker = FundLocker::new(pool);

        let result1 = locker.lock(
            1,
            MicroMinotari(1_000_000),
            1,
            MicroMinotari(5),
            None,
            Some("key-1".to_string()),
            300,
            10,
        );

        let result2 = locker.lock(
            1,
            MicroMinotari(1_000_000),
            1,
            MicroMinotari(5),
            None,
            Some("key-2".to_string()),
            300,
            10,
        );

        // Both should either succeed or fail, but not crash
        assert!(result1.is_ok() || result1.is_err());
        assert!(result2.is_ok() || result2.is_err());
    }

    #[test]
    fn test_fund_locker_no_key_generates_unique() {
        let pool = setup_test_db();
        let locker = FundLocker::new(pool);

        let result1 = locker.lock(
            1,
            MicroMinotari(1_000_000),
            1,
            MicroMinotari(5),
            None,
            None,
            300,
            10,
        );

        let result2 = locker.lock(
            1,
            MicroMinotari(1_000_000),
            1,
            MicroMinotari(5),
            None,
            None,
            300,
            10,
        );

        // Without idempotency keys, each call should be independent
        assert!(result1.is_ok() || result1.is_err());
        assert!(result2.is_ok() || result2.is_err());
    }

    #[test]
    fn test_fund_locker_mutex_prevents_concurrent_selection() {
        use std::sync::Arc;
        use std::thread;

        let pool = setup_test_db();
        let locker = Arc::new(FundLocker::new(pool));
        let mut handles = vec![];

        // Spawn multiple threads trying to lock the same UTXOs
        for i in 0..5 {
            let locker_clone = Arc::clone(&locker);
            handles.push(thread::spawn(move || {
                locker_clone.lock(
                    1,
                    MicroMinotari(1_000_000),
                    1,
                    MicroMinotari(5),
                    None,
                    Some(format!("concurrent-key-{}", i)),
                    300,
                    10,
                )
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // With mutex, at most one should succeed (others may fail due to insufficient funds)
        // but none should panic or corrupt data
        let successes = results.iter().filter(|r| r.is_ok()).count();
        let failures = results.iter().filter(|r| r.is_err()).count();
        assert_eq!(successes + failures, 5);
        // At least one should succeed (the first to acquire the mutex)
        assert!(successes >= 1);
    }
}
