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
    KeyWithTTLCompactionFilter()
        : _value_schema_version(0), _enabled(false), _partition_id(0), _partition_version(0)
    {
    }
    virtual bool Filter(int /*level*/,
                        const rocksdb::Slice &key,
                        const rocksdb::Slice &existing_value,
                        std::string *new_value,
                        bool *value_changed) const override
    {
        if (!_enabled.load(std::memory_order_acquire))
            return false;

        // if value expired, return true immediately
        if (check_if_record_expired(
                _value_schema_version, utils::epoch_now(), utils::to_string_view(existing_value))) {
            return true;
        }

        // _partition_version < 0 indicate current partition not in service
        // if current partition not served this key, return true immediately
        if (_partition_version > 0 && _partition_id <= _partition_version) {
            if(key.size() < 2){
                return true;
            }else{
                uint32_t hash_num = (uint32_t)pegasus_key_hash(key);
                if ((hash_num & _partition_version) != _partition_id) {
                    ddebug("this value will be removed, hash_num is %d, _partition_version=%d, _partition_id=%d",
                           hash_num,
                           _partition_version.load(),
                           _partition_id);
                    return true;
                }
            }
        }
        return false;
    }
    virtual const char *Name() const override { return "KeyWithTTLCompactionFilter"; }
    void SetValueSchemaVersion(uint32_t version) { _value_schema_version = version; }
    void EnableFilter() { _enabled.store(true, std::memory_order_release); }
    void SetPartitionId(uint32_t partition_id) { _partition_id = partition_id; }
    void SetPartitionVersion(uint32_t partition_version)
    {
        _partition_version.store(partition_version);
    }

private:
    uint32_t _value_schema_version;
    std::atomic_bool _enabled; // only process filtering when _enabled == true
    uint32_t _partition_id;
    std::atomic<uint32_t> _partition_version;
};

class KeyWithTTLCompactionFilterFactory : public rocksdb::CompactionFilterFactory
{
public:
    KeyWithTTLCompactionFilterFactory() {}
    virtual std::unique_ptr<rocksdb::CompactionFilter>
    CreateCompactionFilter(const rocksdb::CompactionFilter::Context & /*context*/) override
    {
        return std::unique_ptr<KeyWithTTLCompactionFilter>(new KeyWithTTLCompactionFilter());
    }
    virtual const char *Name() const override { return "KeyWithTTLCompactionFilterFactory"; }
};
}
} // namespace
