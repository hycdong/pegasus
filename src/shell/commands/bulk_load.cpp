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

    ::dsn::error_code ret =
        sc->ddl_client->start_bulk_load(app_name, cluster_name, file_provider_type);

    if (ret != ::dsn::ERR_OK) {
        fprintf(stderr, "start bulk load failed, err = %s\n", ret.to_string());
    }

    return true;
}
