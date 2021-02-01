// Copyright (c) 2019, Xiaomi, Inc.  All rights reserved.
// This source code is licensed under the Apache License Version 2.0, which
// can be found in the LICENSE file in the root directory of this source tree.

#include "shell/commands.h"
#include <dsn/utility/utils.h>

bool start_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'},
                                           {"new_partition_count", required_argument, 0, 'p'},
                                           {0, 0, 0, 0}};

    std::string app_name;
    int32_t new_partition_count = 0;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:p:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        case 'p':
            new_partition_count = boost::lexical_cast<int32_t>(optarg);
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }

    auto err_resp = sc->ddl_client->start_partition_split(app_name, new_partition_count);
    dsn::error_s err = err_resp.get_error();
    auto resp = err_resp.get_value();

    std::string hint_msg;
    if (err.is_ok()) {
        err = dsn::error_s::make(err_resp.get_value().err);
        hint_msg = resp.hint_msg;
    }
    if (!err.is_ok()) {
        fmt::print(stderr, "start partition split failed, error={} [hint:\"{}\"]\n", err, hint_msg);
    } else {
        fmt::print(stdout, "start split succeed\n");
    }

    return true;
}

bool query_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'}, {0, 0, 0, 0}};

    std::string app_name;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }

    auto err_resp = sc->ddl_client->query_partition_split(app_name);
    dsn::error_s err = err_resp.get_error();
    auto resp = err_resp.get_value();

    std::string hint_msg;
    if (err.is_ok()) {
        err = dsn::error_s::make(err_resp.get_value().err);
        hint_msg = resp.hint_msg;
    }
    if (!err.is_ok()) {
        fmt::print(stderr, "query partition split failed, error={} [hint:\"{}\"]\n", err, hint_msg);
        return true;
    }

    // print query split process
    dsn::utils::multi_table_printer mtp;
    dsn::utils::table_printer tp_progress("progress");
    tp_progress.add_title("pidx");
    tp_progress.add_column("partition_status");
    for (auto i = 0; i < resp.new_partition_count / 2; ++i) {
        tp_progress.add_row(i);
        auto iter = resp.status.find(i);
        tp_progress.append_data(iter == resp.status.end() ? "finish split"
                                                          : dsn::enum_to_string(iter->second));
    }
    mtp.add(std::move(tp_progress));

    dsn::utils::table_printer tp_count("count");
    int32_t splitting_count = resp.status.size();
    tp_count.add_row_name_and_data("splitting_count", splitting_count);
    tp_count.add_row_name_and_data("finish_split_count",
                                   resp.new_partition_count / 2 - splitting_count);
    mtp.add(std::move(tp_count));
    mtp.output(std::cout, tp_output_format::kTabular);

    return true;
}

bool pause_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'},
                                           {"parent_partition_index", required_argument, 0, 'i'},
                                           {0, 0, 0, 0}};

    std::string app_name;
    int32_t pidx = -1;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:i:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        case 'i':
            pidx = boost::lexical_cast<int32_t>(optarg);
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }

    auto err_resp = sc->ddl_client->pause_partition_split(app_name, pidx);
    dsn::error_s err = err_resp.get_error();
    auto resp = err_resp.get_value();

    std::string hint_msg;
    if (err.is_ok()) {
        err = dsn::error_s::make(err_resp.get_value().err);
        hint_msg = resp.hint_msg;
    }
    if (!err.is_ok()) {
        fmt::print(stderr, "pause partition split failed, error={} [hint:\"{}\"]\n", err, hint_msg);
    } else {
        fmt::print(stdout, "pause split succeed\n");
    }

    return true;
}

bool restart_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'},
                                           {"parent_partition_index", required_argument, 0, 'i'},
                                           {0, 0, 0, 0}};

    std::string app_name;
    int32_t pidx = -1;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:i:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        case 'i':
            pidx = boost::lexical_cast<int32_t>(optarg);
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }

    auto err_resp = sc->ddl_client->restart_partition_split(app_name, pidx);
    dsn::error_s err = err_resp.get_error();
    auto resp = err_resp.get_value();

    std::string hint_msg;
    if (err.is_ok()) {
        err = dsn::error_s::make(err_resp.get_value().err);
        hint_msg = resp.hint_msg;
    }
    if (!err.is_ok()) {
        fmt::print(
            stderr, "restart partition split failed, error={} [hint:\"{}\"]\n", err, hint_msg);
    } else {
        fmt::print(stdout, "restart split succeed\n");
    }

    return true;
}

bool cancel_partition_split(command_executor *e, shell_context *sc, arguments args)
{
    static struct option long_options[] = {{"app_name", required_argument, 0, 'a'},
                                           {"old_partition_count", required_argument, 0, 'p'},
                                           {0, 0, 0, 0}};

    std::string app_name;
    int32_t old_partition_count = 0;

    optind = 0;
    while (true) {
        int option_index = 0;
        int c;
        c = getopt_long(args.argc, args.argv, "a:p:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            app_name = optarg;
            break;
        case 'p':
            old_partition_count = boost::lexical_cast<int32_t>(optarg);
            break;
        default:
            return false;
        }
    }

    if (app_name.empty()) {
        fprintf(stderr, "app_name should not be empty\n");
        return false;
    }

    auto err_resp = sc->ddl_client->cancel_partition_split(app_name, old_partition_count);
    dsn::error_s err = err_resp.get_error();
    auto resp = err_resp.get_value();

    std::string hint_msg;
    if (err.is_ok()) {
        err = dsn::error_s::make(err_resp.get_value().err);
        hint_msg = resp.hint_msg;
    }
    if (!err.is_ok()) {
        fmt::print(
            stderr, "cancel partition split failed, error={} [hint:\"{}\"]\n", err, hint_msg);
    } else {
        fmt::print(stdout, "cancel split succeed\n");
    }

    return true;
}
