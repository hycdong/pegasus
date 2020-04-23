// Copyright (c) 2019, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include "shell/commands.h"

bool start_bulk_load(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'},
                                           {"cluster_name", required_argument, 0, 'c'},
                                           {"file_provider_type", required_argument, 0, 'p'},
                                           {0, 0, 0, 0}};
    std::string app_name;
    std::string cluster_name;
    std::string file_provider_type;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:c:p:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        case 'c':
            cluster_name = optarg;
            break;
        case 'p':
            file_provider_type = optarg;
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }
    if (cluster_name.empty()) {
        fprintf(stderr, "cluster_name should not be empty\n");
        return false;
    }

    if (file_provider_type.empty()) {
        fprintf(stderr, "file_provider_type should not be empty\n");
        return false;
    }

    auto err_resp = sc->ddl_client->start_bulk_load(app_name, cluster_name, file_provider_type);
    dsn::error_s err = err_resp.get_error();
    std::string hint_msg;
    if (err.is_ok()) {
        err = dsn::error_s::make(err_resp.get_value().err);
        hint_msg = err_resp.get_value().hint_msg;
    }
    if (!err.is_ok()) {
        fmt::print(stderr, "start bulk load failed, error={} [hint:\"{}\"]\n", err, hint_msg);
    } else {
        fmt::print(stdout, "start bulk load succeed\n");
    }

    return true;
}

bool query_bulk_load_status(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'},
                                           {"partition_index", required_argument, 0, 'i'},
                                           {"detailed", no_argument, 0, 'd'},
                                           {0, 0, 0, 0}};

    std::string app_name;
    int32_t pidx = -1;
    bool detailed = false;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:i:d", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        case 'i':
            pidx = boost::lexical_cast<int32_t>(optarg);
            break;
        case 'd':
            detailed = true;
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }

    ::dsn::error_code ret = sc->ddl_client->query_bulk_load(app_name, pidx, detailed);

    if (ret != ::dsn::ERR_OK) {
        fprintf(stderr, "query bulk load status failed, err = %s\n", ret.to_string());
    }
    return true;
}

bool pause_bulk_load(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_id", required_argument, 0, 'a'}, {0, 0, 0, 0}};
    int32_t app_id = 0;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_id = boost::lexical_cast<int32_t>(optarg);
            break;
        default:
            return false;
        }
    }

    if (app_id <= 0) {
        fprintf(stderr, "app_id should not be greater than zero\n");
        return false;
    }

    ::dsn::error_code ret = sc->ddl_client->control_bulk_load(
        app_id, ::dsn::replication::bulk_load_control_type::BLC_PAUSE);

    if (ret != ::dsn::ERR_OK) {
        fprintf(stderr, "pause bulk load failed, err = %s\n", ret.to_string());
    }
    return true;
}

bool restart_bulk_load(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_id", required_argument, 0, 'a'}, {0, 0, 0, 0}};
    int32_t app_id = 0;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_id = boost::lexical_cast<int32_t>(optarg);
            break;
        default:
            return false;
        }
    }

    if (app_id <= 0) {
        fprintf(stderr, "app_id should not be greater than zero\n");
        return false;
    }

    ::dsn::error_code ret = sc->ddl_client->control_bulk_load(
        app_id, ::dsn::replication::bulk_load_control_type::BLC_RESTART);

    if (ret != ::dsn::ERR_OK) {
        fprintf(stderr, "restart bulk load failed, err = %s\n", ret.to_string());
    }
    return true;
}

bool cancel_bulk_load(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {
        {"app_id", required_argument, 0, 'a'}, {"forced", no_argument, 0, 'f'}, {0, 0, 0, 0}};
    int32_t app_id = 0;
    bool forced = false;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:f", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_id = boost::lexical_cast<int32_t>(optarg);
            break;
        case 'f':
            forced = true;
            break;
        default:
            return false;
        }
    }

    if (app_id <= 0) {
        fprintf(stderr, "app_id should not be greater than zero\n");
        return false;
    }

    auto type = forced ? ::dsn::replication::bulk_load_control_type::BLC_FORCE_CANCEL
                       : ::dsn::replication::bulk_load_control_type::BLC_CANCEL;
    ::dsn::error_code ret = sc->ddl_client->control_bulk_load(app_id, type);

    if (ret != ::dsn::ERR_OK) {
        fprintf(stderr, "cancel bulk load failed, err = %s\n", ret.to_string());
    }
    return true;
}
