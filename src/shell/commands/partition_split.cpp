// Copyright (c) 2019, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include "shell/commands.h"

bool app_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    if (args.argc < 3)
        return false;

    std::string app_name = args.argv[1];
    int partition_count = atoi(args.argv[2]);
    if (partition_count <= 0) {
        fprintf(stderr, "new partition count should be positive integer");
        return false;
    }

    ::dsn::error_code err = sc->ddl_client->app_partition_split(app_name, partition_count);
    if (err == ::dsn::ERR_OK)
        std::cout << "split app " << app_name << " succeed" << std::endl;
    else
        std::cout << "split app " << app_name << " failed, error=" << err.to_string() << std::endl;

    return true;
}

bool pause_single_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    if (args.argc < 3)
        return false;

    std::string app_name = args.argv[1];
    int pause_partition_index = atoi(args.argv[2]);

    if (pause_partition_index < 0) {
        fprintf(stderr, "partition index should be greater than zero");
        return false;
    }

    // ddl->pause_single_partition
    ::dsn::error_code err =
        sc->ddl_client->control_single_partition_split(app_name, pause_partition_index, true);
    if (err == ::dsn::ERR_OK || err == ::dsn::ERR_NO_NEED_OPERATE)
        std::cout << "pause split app " << app_name << " partition[" << pause_partition_index
                  << "] succeed" << std::endl;
    else
        std::cout << "pause split app " << app_name << " partition[" << pause_partition_index
                  << "] failed, error is " << err.to_string() << std::endl;

    return true;
}

bool restart_single_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    if (args.argc < 3)
        return false;

    std::string app_name = args.argv[1];
    int restart_partition_index = atoi(args.argv[2]);

    if (restart_partition_index < 0) {
        fprintf(stderr, "partition index should be greater than zero");
        return false;
    }

    // ddl->pause_single_partition
    ::dsn::error_code err =
        sc->ddl_client->control_single_partition_split(app_name, restart_partition_index, false);
    if (err == ::dsn::ERR_OK || err == ::dsn::ERR_NO_NEED_OPERATE)
        std::cout << "restart split app " << app_name << " partition[" << restart_partition_index
                  << "] succeed" << std::endl;
    else
        std::cout << "restart split app " << app_name << " partition[" << restart_partition_index
                  << "] failed, error is " << err.to_string() << std::endl;

    return true;
}

bool cancel_app_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    if (args.argc < 3)
        return false;

    static struct option long_options[] = {{"force", no_argument, 0, 'f'}, {0, 0, 0, 0}};

    std::string app_name = args.argv[1];
    int original_partition_count = atoi(args.argv[2]);
    bool forced = false;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "f", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'f':
            forced = true;
            break;
        default:
            return false;
        }
    }

    if (original_partition_count < 0) {
        fprintf(stderr, "partition index should be greater than zero");
        return false;
    }

    // ddl->cancel_app_partition_split
    ::dsn::error_code err =
        sc->ddl_client->cancel_app_partition_split(app_name, original_partition_count, forced);
    if (err == ::dsn::ERR_OK)
        std::cout << "cancel split app " << app_name << " succeed" << std::endl;
    else
        std::cout << "cancel split app " << app_name << " failed, error is " << err.to_string()
                  << std::endl;

    return true;
}

bool clear_partition_split_flag(command_executor *e, shell_context *sc, arguments args)
{
    if (args.argc < 2)
        return false;

    std::string app_name = args.argv[1];
    ::dsn::error_code err = sc->ddl_client->clear_app_split_flags(app_name);
    if (err == ::dsn::ERR_OK)
        std::cout << "clear split flags of app " << app_name << " succeed" << std::endl;
    else
        std::cout << "clear split flags of app " << app_name << " failed, error is "
                  << err.to_string() << std::endl;

    return true;
}
