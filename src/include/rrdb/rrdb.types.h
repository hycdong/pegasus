#pragma once
#include <dsn/service_api_cpp.h>
#include <dsn/cpp/serialization.h>


#include "rrdb_types.h"


namespace dsn { namespace apps { 
    GENERATED_TYPE_SERIALIZATION(update_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(update_response, THRIFT)
    GENERATED_TYPE_SERIALIZATION(read_response, THRIFT)
    GENERATED_TYPE_SERIALIZATION(ttl_response, THRIFT)
    GENERATED_TYPE_SERIALIZATION(count_response, THRIFT)
    GENERATED_TYPE_SERIALIZATION(key_value, THRIFT)
    GENERATED_TYPE_SERIALIZATION(multi_put_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(multi_remove_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(multi_remove_response, THRIFT)
    GENERATED_TYPE_SERIALIZATION(multi_get_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(multi_get_response, THRIFT)
    GENERATED_TYPE_SERIALIZATION(incr_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(get_scanner_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(scan_request, THRIFT)
    GENERATED_TYPE_SERIALIZATION(scan_response, THRIFT)

} } 