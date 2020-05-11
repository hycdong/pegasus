// Copyright (c) 2017, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include <boost/lexical_cast.hpp>

#include <dsn/service_api_c.h>
#include <dsn/dist/replication/replication_ddl_client.h>
#include <dsn/utility/filesystem.h>

#include <pegasus/client.h>
#include <gtest/gtest.h>

#include "base/pegasus_const.h"
#include "global_env.h"

using namespace ::dsn;
using namespace ::dsn::replication;
using namespace pegasus;

class bulk_load_test : public testing::Test
{
protected:
    virtual void SetUp()
    {
        pegasus_root_dir = global_env::instance()._pegasus_root;
        working_root_dir = global_env::instance()._working_dir;
        bulk_load_provider_root =
            dsn::utils::filesystem::path_combine(pegasus_root_dir, LOCAL_ROOT);

        // copy bulk_load files
        copy_bulk_load_files();

        // update config and restart onebox
        system("./run.sh clear_onebox");

        system("cp src/server/config.min.ini config-server-test-bulk-load.ini");
        std::string cmd = "sed -i \"/^\\s*bulk_load_provider_root/c bulk_load_provider_root = " +
                          bulk_load_provider_root;
        cmd = cmd + std::string("\" config-server-test-bulk-load.ini");
        system(cmd.c_str());

        system("./run.sh start_onebox -w --config_path config-server-test-bulk-load.ini");
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // initialize the clients
        std::vector<dsn::rpc_address> meta_list;
        replica_helper::load_meta_servers(
            meta_list, PEGASUS_CLUSTER_SECTION_NAME.c_str(), "mycluster");

        ddl_client = std::make_shared<replication_ddl_client>(meta_list);
        pg_client = pegasus::pegasus_client_factory::get_client("mycluster", APP_NAME.c_str());
    }

    virtual void TearDown()
    {
        chdir(pegasus_root_dir.c_str());
        std::string remove_cmd = "rm -rf " + bulk_load_provider_root;
        system(remove_cmd.c_str());
        system("./run.sh clear_onebox");
        system("./run.sh start_onebox -w");
        chdir(working_root_dir.c_str());
    }

public:
    std::shared_ptr<replication_ddl_client> ddl_client;
    pegasus::pegasus_client *pg_client;
    std::string pegasus_root_dir;
    std::string working_root_dir;
    std::string bulk_load_provider_root;
    enum operation
    {
        GET,
        SET,
        DEL,
        NO_VALUE
    };

public:
    error_code start_bulk_load()
    {
        auto err_resp = ddl_client->start_bulk_load(APP_NAME, CLUSTER, PROVIDER);
        return err_resp.get_value().err;
    }

    void wait_bulk_load_finish(int64_t seconds)
    {
        int64_t sleep_time = 5;
        error_code err = ERR_OK;

        while (seconds > 0 && err == ERR_OK) {
            sleep_time = sleep_time > seconds ? seconds : sleep_time;
            seconds -= sleep_time;
            std::cout << "sleep " << sleep_time << "s to query bulk status" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(sleep_time));

            err = ddl_client->query_bulk_load(APP_NAME).get_value().err;
        }
    }

    bool verify_bulk_load_data()
    {
        if (!verify_data("hashkey", "sortkey")) {
            return false;
        }
        return verify_data(HASHKEY_PREFIX, SORTKEY_PREFIX);
    }

    bool verify_data(const std::string &hashkey_prefix, const std::string &sortkey_prefix)
    {
        const std::string &expected_value = VALUE;
        for (int i = 0; i < COUNT; ++i) {
            std::string hash_key = hashkey_prefix + boost::lexical_cast<std::string>(i);
            for (int j = 0; j < COUNT; ++j) {
                std::string sort_key = sortkey_prefix + boost::lexical_cast<std::string>(j);
                std::string act_value;
                int ret = pg_client->get(hash_key, sort_key, act_value);
                if (ret != 0) {
                    std::cout << "Failed to get [" << hash_key << "," << sort_key << "], error is "
                              << ret << std::endl;
                    return false;
                }
                if (act_value != expected_value) {
                    std::cout << "get [" << hash_key << "," << sort_key
                              << "], value = " << act_value
                              << ", but expected_value = " << expected_value << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    void operate_data(bulk_load_test::operation op, const std::string &value, int count)
    {
        for (int i = 0; i < count; ++i) {
            std::string hash_key = HASHKEY_PREFIX + boost::lexical_cast<std::string>(i);
            std::string sort_key = SORTKEY_PREFIX + boost::lexical_cast<std::string>(i);
            switch (op) {
            case bulk_load_test::operation::GET: {
                std::string act_value;
                int ret = pg_client->get(hash_key, sort_key, act_value);
                ASSERT_EQ(ret, 0);
                ASSERT_EQ(act_value, value);
            } break;
            case bulk_load_test::operation::DEL: {
                int ret = pg_client->del(hash_key, sort_key);
                ASSERT_EQ(0, ret);
            } break;
            case bulk_load_test::operation::SET: {
                int ret = pg_client->set(hash_key, sort_key, value);
                ASSERT_EQ(0, ret);
            } break;
            case bulk_load_test::operation::NO_VALUE: {
                std::string act_value;
                int ret = pg_client->get(hash_key, sort_key, act_value);
                // -1001 means value not found
                ASSERT_EQ(ret, -1001);
            } break;
            default:
                break;
            }
        }
    }

    void copy_bulk_load_files()
    {
        chdir(pegasus_root_dir.c_str());
        std::string copy_file_cmd =
            "cp -r src/test/function_test/bulk_load_files/" + LOCAL_ROOT + " .";
        system(copy_file_cmd.c_str());
    }

    void replace_bulk_load_info()
    {
        chdir(pegasus_root_dir.c_str());
        std::string cmd = "cp -R src/test/function_test/bulk_load_files/mock_bulk_load_info/. " +
                          bulk_load_provider_root + "/" + CLUSTER + "/" + APP_NAME + "/";
        system(cmd.c_str());
    }

    void remove_file(bool is_bulk_load_info)
    {
        std::string file_name = is_bulk_load_info ? "/bulk_load_info" : "/0/bulk_load_metadata";
        std::string cmd =
            "rm " + bulk_load_provider_root + "/" + CLUSTER + "/" + APP_NAME + file_name;
        system(cmd.c_str());
    }

    const std::string LOCAL_ROOT = "bulk_load_test";
    const std::string APP_NAME = "temp";
    const std::string CLUSTER = "cluster";
    const std::string PROVIDER = "local_service";

    const std::string HASHKEY_PREFIX = "hash";
    const std::string SORTKEY_PREFIX = "sort";
    const std::string VALUE = "newValue";
    const int32_t COUNT = 1000;
};

TEST_F(bulk_load_test, failed_bulk_load_info)
{
    // bulk load failed because {bulk_load_info} file is missing
    remove_file(true);
    ASSERT_EQ(start_bulk_load(), ERR_OBJECT_NOT_FOUND);

    // bulk load failed because {bulk_load_info} file inconsistent with actual app info
    replace_bulk_load_info();
    ASSERT_EQ(start_bulk_load(), ERR_INCONSISTENT_STATE);
}

TEST_F(bulk_load_test, bulk_load_after_failed)
{
    // bulk load failed because {bulk_load_metadata} file is missing
    remove_file(false);
    ASSERT_EQ(start_bulk_load(), ERR_OK);
    wait_bulk_load_finish(300);

    // recover complete files
    copy_bulk_load_files();

    ASSERT_EQ(start_bulk_load(), ERR_OK);
    wait_bulk_load_finish(300);
    std::cout << "Start to verify data..." << std::endl;
    ASSERT_TRUE(verify_bulk_load_data());
}

TEST_F(bulk_load_test, double_bulk_load)
{
    ASSERT_EQ(start_bulk_load(), ERR_OK);
    wait_bulk_load_finish(300);
    ASSERT_TRUE(verify_bulk_load_data());

    std::cout << "bulk load twice" << std::endl;

    ASSERT_EQ(start_bulk_load(), ERR_OK);
    wait_bulk_load_finish(300);
    ASSERT_TRUE(verify_bulk_load_data());
}

TEST_F(bulk_load_test, bulk_load_data_consistency)
{
    // write old data
    operate_data(operation::SET, "oldValue", 10);

    ASSERT_EQ(start_bulk_load(), ERR_OK);
    wait_bulk_load_finish(300);
    // value overide by bulk-load-data
    operate_data(operation::GET, VALUE, 10);

    // write data again
    operate_data(operation::SET, "valueAfterBulkLoad", 20);
    operate_data(operation::GET, "valueAfterBulkLoad", 20);

    // del data
    operate_data(operation::DEL, "", 15);
    operate_data(operation::NO_VALUE, "", 15);
}
