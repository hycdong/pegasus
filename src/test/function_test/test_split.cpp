// Copyright (c) 2017, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <pegasus/client.h>
#include <boost/lexical_cast.hpp>

#include <dsn/dist/replication/replication_ddl_client.h>

using namespace dsn::replication;

TEST(split, basic_split)
{
    const std::string split_table = "split_table";
    const int partition_count = 4;

    std::vector<dsn::rpc_address> meta_list;
    replica_helper::load_meta_servers(meta_list, "uri-resolver.dsn://mycluster", "arguments");
    replication_ddl_client *ddl_client = new replication_ddl_client(meta_list);

    // first create table
    std::cerr << "create app " << split_table << std::endl;
    dsn::error_code error =
        ddl_client->create_app(split_table, "pegasus", partition_count, 3, {}, false);
    ASSERT_EQ(dsn::ERR_OK, error);

    pegasus::pegasus_client *pg_client =
        pegasus::pegasus_client_factory::get_client("mycluster", split_table.c_str());

    // write data
    int count = 1000, i;
    for (i = 0; i < count; ++i) {
        std::string hash_key = "hashkey" + boost::lexical_cast<std::string>(i);
        std::string sort_key = "sortkey" + boost::lexical_cast<std::string>(i);
        auto ret = pg_client->set(hash_key, sort_key, "value");
        ASSERT_EQ(ret, 0);
    }

    // execute partition split
    error = ddl_client->app_partition_split(split_table, partition_count * 2);
    ASSERT_EQ(dsn::ERR_OK, error);
}
