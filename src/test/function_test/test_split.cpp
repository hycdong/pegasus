// Copyright (c) 2017, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <pegasus/client.h>
#include <boost/lexical_cast.hpp>

#include <dsn/dist/replication/replication_ddl_client.h>

#include "base/pegasus_const.h"

using namespace dsn::replication;
using namespace pegasus;

int partition_count = 4;
extern replication_ddl_client *ddl_client;

static void create_table(std::string table_name, int partition_count)
{
    std::cerr << "create app " << table_name << std::endl;
    dsn::error_code error =
        ddl_client->create_app(table_name, "pegasus", partition_count, 3, {}, false);
    ASSERT_EQ(dsn::ERR_OK, error);
}

static void drop_table(std::string table_name)
{
    std::cerr << "drop app " << table_name << std::endl;
    dsn::error_code error = ddl_client->drop_app(table_name, 1);
    ASSERT_EQ(dsn::ERR_OK, error);
}

class split : public testing::Test
{
public:
    static void SetUpTestCase()
    {
        ddebug("SetUp...");
        std::vector<dsn::rpc_address> meta_list;
        replica_helper::load_meta_servers(
            meta_list, PEGASUS_CLUSTER_SECTION_NAME.c_str(), "mycluster");
        ddl_client = new replication_ddl_client(meta_list);

        create_table("split_table", partition_count);
        create_table("scan_split", partition_count);
        create_table("pause_split", partition_count);
        create_table("cancel_split", partition_count);
    }

    static void TearDownTestCase()
    {
        ddebug("TearDown...");
        drop_table("split_table");
        drop_table("scan_split");
        drop_table("pause_split");
        drop_table("cancel_split");
    }
};

TEST_F(split, basic_split)
{
    static std::map<std::string, std::map<std::string, std::string>> expected;
    const std::string split_table = "split_table";
    dsn::error_code error;
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
    while (!completed) {
        int app_id, partition;
        std::vector<dsn::partition_configuration> partitions;
        error = ddl_client->list_app(split_table, app_id, partition, partitions);
        ASSERT_EQ(dsn::ERR_OK, error);
        ASSERT_EQ(partition, partition_count * 2);

        completed = true;
        for (int i = 0; i < partition; ++i) {
            if (partitions[i].ballot == invalid_ballot) {
                completed = false;
                break;
            }
        }

        if (!completed) {
            int ret;
            std::string hash = "keyh_" + boost::lexical_cast<std::string>(counter);
            std::string sort = "keys_" + boost::lexical_cast<std::string>(counter);

            ret = pg_client->set(hash, sort, "value");
            if (ret == 0) {
                expected[hash][sort] = "value";
                std::cerr << "set " << counter << " succeed" << std::endl;
                counter++;
            } else {
                std::cerr << "set " << counter << " failed, error is "
                          << pg_client->get_error_string(ret) << std::endl;
            }
            // ok or timeout
            ASSERT_TRUE((ret == 0 || ret == -2));

            std::cerr << "split is not finished, wait for 1 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            std::cerr << "partition split finished" << std::endl;
        }
    }

    // validate data after partition split
    for (i = 0; i < count; ++i) {
        std::string hash_key = "hashkey" + boost::lexical_cast<std::string>(i);
        std::string sort_key = "sortkey" + boost::lexical_cast<std::string>(i);
        std::string value;
        auto ret = pg_client->get(hash_key, sort_key, value);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(expected[hash_key][sort_key], value);
    }

    for (i = 0; i < counter; ++i) {
        std::string hash_key = "keyh_" + boost::lexical_cast<std::string>(i);
        std::string sort_key = "keys_" + boost::lexical_cast<std::string>(i);
        std::string value;
        auto ret = pg_client->get(hash_key, sort_key, value);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(expected[hash_key][sort_key], value);
    }
}

TEST_F(split, split_scan)
{
    static std::map<std::string, std::map<std::string, std::string>> expected;
    static std::map<std::string, std::map<std::string, std::string>> actual;
    const std::string split_table = "scan_split";
    dsn::error_code error;

    pegasus::pegasus_client *pg_client =
        pegasus::pegasus_client_factory::get_client("mycluster", split_table.c_str());

    // write data
    int count = 1000, i;
    std::string hash_key = "hashkey";
    for (i = 0; i < count; ++i) {
        std::string sort_key = "sortkey" + boost::lexical_cast<std::string>(i);
        auto ret = pg_client->set(hash_key, sort_key, "value");
        expected[hash_key][sort_key] = "value";
        ASSERT_EQ(ret, 0);
    }

    // succeed
    error = ddl_client->app_partition_split(split_table, partition_count * 2);
    ASSERT_EQ(dsn::ERR_OK, error);

    // wait split finish
    int try_count = 0;
    bool completed = false;
    while (!completed) {
        int app_id, partition;
        std::vector<dsn::partition_configuration> partitions;
        error = ddl_client->list_app(split_table, app_id, partition, partitions);
        ASSERT_EQ(dsn::ERR_OK, error);
        ASSERT_EQ(partition, partition_count * 2);

        completed = true;
        for (int i = 0; i < partition; ++i) {
            if (partitions[i].ballot == invalid_ballot) {
                completed = false;
                break;
            }
        }
        if (!completed) {
            if ((++try_count) % 3 == 0) {
                std::cerr << "try hash_scan during split, try count is " << try_count / 3
                          << std::endl;
                // hash_scan during this procedure
                pegasus::pegasus_client::pegasus_scanner *scanner = nullptr;
                pegasus::pegasus_client::scan_options options;
                int ret = pg_client->get_scanner(hash_key, "", "", options, scanner);
                ASSERT_EQ(ret, 0);

                // verify scan value
                std::string hash_temp;
                std::string sort_temp;
                std::string value_temp;
                int scan_count = 0;
                while (scanner->next(hash_temp, sort_temp, value_temp) == 0) {
                    scan_count++;
                    ASSERT_EQ(expected[hash_temp][sort_temp], value_temp);
                }
                ASSERT_EQ(scan_count, count);
                delete scanner;
            }
            std::cerr << "split is not finished, wait for 1 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            std::cerr << "partition split finished" << std::endl;
        }
    }

    // validate data after partition split
    for (i = 0; i < count; ++i) {
        std::string hash_key = "hashkey";
        std::string sort_key = "sortkey" + boost::lexical_cast<std::string>(i);
        std::string value;
        auto ret = pg_client->get(hash_key, sort_key, value);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(expected[hash_key][sort_key], value);
    }
}

TEST_F(split, pause_split)
{
    static std::map<std::string, std::map<std::string, std::string>> expected;
    const std::string split_table = "pause_split";
    dsn::error_code error;
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

    // succeed
    error = ddl_client->app_partition_split(split_table, partition_count * 2);
    ASSERT_EQ(dsn::ERR_OK, error);

    // set during this procedure
    int counter = 0;
    bool completed = false;
    while (!completed) {
        // pause partition[1]
        if (counter == 1) {
            // [keyh_1, keys_1, value] should be insert into partition 1
            error = ddl_client->control_single_partition_split(split_table, 1, true);
            ASSERT_EQ(dsn::ERR_OK, error);
        }
        if (counter == 30) {
            // restart split
            error = ddl_client->control_single_partition_split(split_table, 1, false);
            ASSERT_EQ(dsn::ERR_OK, error);
        }

        int app_id, partition;
        std::vector<dsn::partition_configuration> partitions;
        error = ddl_client->list_app(split_table, app_id, partition, partitions);
        ASSERT_EQ(dsn::ERR_OK, error);
        ASSERT_EQ(partition, partition_count * 2);

        completed = true;
        for (int i = 0; i < partition; ++i) {
            if (partitions[i].ballot == invalid_ballot) {
                completed = false;
                break;
            }
        }

        if (!completed) {
            int ret;
            std::string hash = "keyh_" + boost::lexical_cast<std::string>(counter);
            std::string sort = "keys_" + boost::lexical_cast<std::string>(counter);

            ret = pg_client->set(hash, sort, "value");
            if (ret == 0) {
                expected[hash][sort] = "value";
                std::cerr << "set " << counter << " succeed" << std::endl;
                counter++;
            } else {
                std::cerr << "set " << counter << " failed, error is "
                          << pg_client->get_error_string(ret) << std::endl;
            }
            // ok or timeout
            ASSERT_TRUE((ret == 0 || ret == -2));

            std::cerr << "split is not finished, wait for 1 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            std::cerr << "partition split finished" << std::endl;
        }
    }

    // validate data after partition split
    for (i = 0; i < count; ++i) {
        std::string hash_key = "hashkey" + boost::lexical_cast<std::string>(i);
        std::string sort_key = "sortkey" + boost::lexical_cast<std::string>(i);
        std::string value;
        auto ret = pg_client->get(hash_key, sort_key, value);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(expected[hash_key][sort_key], value);
    }

    for (i = 0; i < counter; ++i) {
        std::string hash_key = "keyh_" + boost::lexical_cast<std::string>(i);
        std::string sort_key = "keys_" + boost::lexical_cast<std::string>(i);
        std::string value;
        auto ret = pg_client->get(hash_key, sort_key, value);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(expected[hash_key][sort_key], value);
    }
}

TEST_F(split, cancel_split)
{
    static std::map<std::string, std::map<std::string, std::string>> expected;
    const std::string split_table = "cancel_split";
    dsn::error_code error;
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

    // succeed
    error = ddl_client->app_partition_split(split_table, partition_count * 2);
    ASSERT_EQ(dsn::ERR_OK, error);

    // set during this procedure
    int counter = 0;
    bool completed = false;
    while (!completed) {
        // pause partition[0]
        if (counter == 1) {
            error = ddl_client->control_single_partition_split(split_table, 0, true);
            ASSERT_EQ(dsn::ERR_OK, error);
            std::cerr << "pause split partition[0]" << std::endl;
        }

        int app_id, partition;
        std::vector<dsn::partition_configuration> partitions;
        error = ddl_client->list_app(split_table, app_id, partition, partitions);
        ASSERT_EQ(dsn::ERR_OK, error);
        ASSERT_EQ(partition, partition_count * 2);

        int finish_count = 0;
        for (int i = partition_count; i < partition; ++i) {
            if (partitions[i].ballot > invalid_ballot) {
                ++finish_count;
            }
        }

        if (finish_count == partition_count - 1) {
            completed = true;
        } else {
            std::cerr << (partition_count - finish_count) << " partition not finish split"
                      << std::endl;
        }
        ++counter;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // [0,y,z] will write to partition[3], won't lost
    // [5,y,z] will write to partition[5], will be lost after cancel
    std::string hash_key1 = boost::lexical_cast<std::string>(0);
    std::string hash_key2 = boost::lexical_cast<std::string>(5);
    for (i = 0; i < counter; ++i) {
        std::string sort_key = boost::lexical_cast<std::string>(i);
        auto ret = pg_client->set(hash_key1, sort_key, "value");
        ASSERT_EQ(ret, 0);
        expected[hash_key1][sort_key] = "value";

        ret = pg_client->set(hash_key2, sort_key, "value");
        ASSERT_EQ(ret, 0);
    }

    // cancel partition split
    error = ddl_client->cancel_app_partition_split(split_table, partition_count, true);
    ASSERT_EQ(dsn::ERR_OK, error);
    std::cerr << "cancel partition split" << std::endl;

    std::cerr << "sleep for 30 seconds to notify replica server" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(30));

    // validate data after partition split
    while (i < count) {
        std::string hash_key = "hashkey" + boost::lexical_cast<std::string>(i);
        std::string sort_key = "sortkey" + boost::lexical_cast<std::string>(i);
        std::string value;
        auto ret = pg_client->get(hash_key, sort_key, value);
        // retry when timeout
        if (ret == -2) {
            continue;
        } else {
            ASSERT_EQ(expected[hash_key][sort_key], value);
            ++i;
        }
    }

    for (i = 0; i < counter; ++i) {
        // [0,y,z] won't be lost
        std::string sort_key = boost::lexical_cast<std::string>(i);
        std::string value1;
        auto ret = pg_client->get(hash_key1, sort_key, value1);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(expected[hash_key1][sort_key], value1);

        // [5,y,z] will be lost
        std::string value2;
        ret = pg_client->get(hash_key2, sort_key, value2);
        // not found
        ASSERT_EQ(ret, -1001);
        ASSERT_EQ("", value2);
    }
}
