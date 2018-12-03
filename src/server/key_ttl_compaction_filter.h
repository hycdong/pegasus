// Copyright (c) 2017, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <cinttypes>
#include <atomic>
#include <rocksdb/compaction_filter.h>
#include <rocksdb/merge_operator.h>

#include "base/pegasus_utils.h"
#include "base/pegasus_value_schema.h"
#include "base/pegasus_key_schema.h"

namespace pegasus {
namespace server {

class KeyWithTTLCompactionFilter : public rocksdb::CompactionFilter
{
public:
    KeyWithTTLCompactionFilter(uint32_t value_schema_version,
                               uint32_t default_ttl,
                               bool enabled,
                               uint32_t partition_id,
                               uint32_t partition_version)
        : _value_schema_version(value_schema_version),
          _default_ttl(default_ttl),
          _enabled(enabled),
          _partition_id(partition_id),
          _partition_version(partition_version)

    {
    }

    bool Filter(int /*level*/,
                const rocksdb::Slice &key,
                const rocksdb::Slice &existing_value,
                std::string *new_value,
                bool *value_changed) const override
    {
        if (!_enabled) {
            return false;
        }

        uint32_t expire_ts =
            pegasus_extract_expire_ts(_value_schema_version, utils::to_string_view(existing_value));
        if (_default_ttl != 0 && expire_ts == 0) {
            // should update ttl
            *new_value = existing_value.ToString();
            pegasus_update_expire_ts(
                _value_schema_version, *new_value, utils::epoch_now() + _default_ttl);
            *value_changed = true;
            return false;
        }

        if (check_if_ts_expired(utils::epoch_now(), expire_ts)) {
            return true;
        }

        // _partition_version < 0 indicate current partition not in service
        // if current partition not served this key, return true immediately
        if (_partition_version > 0 && _partition_id <= _partition_version) {
            if (key.size() < 2) {
                return true;
            } else {
                uint32_t hash_num = (uint32_t)pegasus_key_hash(key);
                if ((hash_num & _partition_version) != _partition_id) {
                    dinfo("this value will be removed, hash_num is %d, _partition_version=%d, "
                          "_partition_id=%d",
                          hash_num,
                          _partition_version,
                          _partition_id);
                    return true;
                }
            }
        }
        return false;
    }

    const char *Name() const override { return "KeyWithTTLCompactionFilter"; }

private:
    uint32_t _value_schema_version;
    uint32_t _default_ttl;
    bool _enabled; // only process filtering when _enabled == true
    mutable pegasus_value_generator _gen;

    uint32_t _partition_id;
    uint32_t _partition_version;
};

class KeyWithTTLCompactionFilterFactory : public rocksdb::CompactionFilterFactory
{
public:
    KeyWithTTLCompactionFilterFactory() : _value_schema_version(0), _default_ttl(0), _enabled(false)
    {
    }
    std::unique_ptr<rocksdb::CompactionFilter>
    CreateCompactionFilter(const rocksdb::CompactionFilter::Context & /*context*/) override
    {
        return std::unique_ptr<KeyWithTTLCompactionFilter>(
            new KeyWithTTLCompactionFilter(_value_schema_version.load(),
                                           _default_ttl.load(),
                                           _enabled.load(),
                                           _partition_id.load(),
                                           _partition_version.load()));
    }
    const char *Name() const override { return "KeyWithTTLCompactionFilterFactory"; }

    void SetValueSchemaVersion(uint32_t version)
    {
        _value_schema_version.store(version, std::memory_order_release);
    }
    void EnableFilter() { _enabled.store(true, std::memory_order_release); }
    void SetDefaultTTL(uint32_t ttl) { _default_ttl.store(ttl, std::memory_order_release); }
    void SetPartitionId(uint32_t partition_id) { _partition_id.store(partition_id); }
    void SetPartitionVersion(uint32_t partition_version)
    {
        _partition_version.store(partition_version);
    }

private:
    std::atomic<uint32_t> _value_schema_version;
    std::atomic<uint32_t> _default_ttl;
    std::atomic_bool _enabled; // only process filtering when _enabled == true

    std::atomic<uint32_t> _partition_id;
    std::atomic<uint32_t> _partition_version;
};

} // namespace server
} // namespace pegasus
