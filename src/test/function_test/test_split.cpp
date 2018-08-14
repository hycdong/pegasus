// Copyright (c) 2017, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <pegasus/client.h>
#include <boost/lexical_cast.hpp>

#include <pegasus/client.h>
#include <dsn/dist/replication/replication_ddl_client.h>

using namespace dsn::replication;

TEST(split, basic_split)
{
    static std::map<std::string, std::map<std::string, std::string>> expected;
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
        expected[hash_key][sort_key] = "value";
        ASSERT_EQ(ret, 0);
    }

    // wrong partition_count
    error = ddl_client->app_partition_split(split_table, partition_count);
    ASSERT_EQ(dsn::ERR_INVALID_PARAMETERS, error);

    // succeed
    error = ddl_client->app_partition_split(split_table, partition_count * 2);
    ASSERT_EQ(dsn::ERR_OK, error);

    error = ddl_client->app_partition_split(split_table, partition_count * 2);
    ASSERT_EQ(dsn::ERR_INVALID_PARAMETERS, error);

    // busy
    error = ddl_client->app_partition_split(split_table, partition_count * 4);
    ASSERT_EQ(dsn::ERR_BUSY, error);

    // set during this procedure
    int counter = 0;
    bool completed = false;
    while(!completed){
        int app_id, partition;
        std::vector<dsn::partition_configuration> partitions;
        error = ddl_client->list_app(split_table, app_id, partition, partitions);
        ASSERT_EQ(dsn::ERR_OK, error);
        ASSERT_EQ(partition, partition_count*2);

        completed = true;
        for(int i = 0; i < partition; ++i){
            if(partitions[i].ballot == invalid_ballot){
                completed = false;
                break;
            }
        }

        if(!completed){
            int ret;
            std::string hash = "hash_" + boost::lexical_cast<std::string>(counter);
            std::string sort = "sort_" + boost::lexical_cast<std::string>(counter);

            ret = pg_client->set(hash, sort, "value");
            if (ret == 0) {
                expected[hash][sort] = "value";
                counter++;
                std::cerr << "set " << counter-1 << " succeed" << std::endl;
            } else {
                std::cerr << "set " << counter-1 << " failed, error is " << pg_client->get_error_string(ret) << std::endl;
            }
            // ok or timeout
            ASSERT_TRUE((ret == 0 || ret == -2));

            //TODO(hyc): change to 5 seconds will lead to error, check partition_version=-1
            std::cerr << "split is not finished, wait for 5 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }else{
            std::cerr << "partition split finished" << std::endl;
        }
    }
}
