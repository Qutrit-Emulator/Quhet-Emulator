/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * HEXSTATE ENGINE — 6-State Quantum Processor (Entry Point)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Usage:
 *   ./hexstate_engine                    Interactive mode (stub)
 *   ./hexstate_engine <program.qbin>     Execute program
 *   ./hexstate_engine --self-test        Run built-in verification
 *
 * All chunks use Magic Pointers (0x4858 tag) to reference external Hilbert space.
 * Local mmap'd RAM serves as a shadow cache of the external quantum state.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hexstate_engine.h"

static void print_banner(void)
{
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  HEXSTATE ENGINE v1.0\n");
    printf("  6-State Quantum Processor\n");
    printf("  |0⟩ |1⟩ |2⟩ |3⟩ |4⟩ |5⟩\n");
    printf("  Magic Pointer Architecture (Hilbert Tag: 0x4858)\n");
    printf("  BigInt: 4096-bit | Max Chunks: 16.7M\n");
    printf("══════════════════════════════════════════════════════\n\n");
}

static void interactive_loop(HexStateEngine *eng)
{
    printf("Interactive mode. Commands:\n");
    printf("  init <id> <hexits>   - Initialize chunk\n");
    printf("  sup <id>             - Create superposition\n");
    printf("  had <id> <hexit>     - Apply DFT₆ Hadamard\n");
    printf("  meas <id>            - Measure chunk\n");
    printf("  grov <id>            - Grover diffusion\n");
    printf("  braid <a> <b>        - Entangle chunks\n");
    printf("  fork <dst> <src>     - Timeline fork\n");
    printf("  inf <id> [size]      - Infinite resources\n");
    printf("  print <id>           - Print chunk state\n");
    printf("  summary              - Engine summary\n");
    printf("  quit                 - Exit\n\n");

    char line[256];
    while (eng->running) {
        printf("> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        /* Remove trailing newline */
        line[strcspn(line, "\n")] = '\0';
        if (strlen(line) == 0) continue;

        char cmd[32];
        uint64_t arg1 = 0, arg2 = 0;
        int nargs = sscanf(line, "%31s %lu %lu", cmd, &arg1, &arg2);

        if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0) {
            eng->running = 0;
        } else if (strcmp(cmd, "init") == 0 && nargs >= 3) {
            init_chunk(eng, arg1, arg2);
        } else if (strcmp(cmd, "sup") == 0 && nargs >= 2) {
            create_superposition(eng, arg1);
        } else if (strcmp(cmd, "had") == 0 && nargs >= 3) {
            apply_hadamard(eng, arg1, arg2);
        } else if (strcmp(cmd, "meas") == 0 && nargs >= 2) {
            uint64_t result = measure_chunk(eng, arg1);
            printf("  => %lu\n", result);
        } else if (strcmp(cmd, "grov") == 0 && nargs >= 2) {
            grover_diffusion(eng, arg1);
        } else if (strcmp(cmd, "braid") == 0 && nargs >= 3) {
            braid_chunks(eng, arg1, arg2, 0, 0);
        } else if (strcmp(cmd, "fork") == 0 && nargs >= 3) {
            op_timeline_fork(eng, arg1, arg2);
        } else if (strcmp(cmd, "inf") == 0 && nargs >= 2) {
            op_infinite_resources(eng, arg1, nargs >= 3 ? arg2 : 0);
        } else if (strcmp(cmd, "print") == 0 && nargs >= 2) {
            print_chunk_state(eng, arg1);
        } else if (strcmp(cmd, "summary") == 0) {
            Instruction instr = {OP_SUMMARY, 0, 0, 0};
            execute_instruction(eng, instr);
        } else {
            printf("  Unknown command: %s\n", cmd);
        }
    }
}

int main(int argc, char *argv[])
{
    HexStateEngine eng;

    print_banner();

    if (engine_init(&eng) != 0) {
        fprintf(stderr, "[FATAL] Engine initialization failed\n");
        return 1;
    }

    if (argc >= 2 && strcmp(argv[1], "--self-test") == 0) {
        int ret = run_self_test(&eng);
        engine_destroy(&eng);
        return ret;
    }

    if (argc >= 3 && strcmp(argv[1], "--shor") == 0) {
        /* argv[2] is the decimal string N — passed as-is, no 64-bit limit */
        int ret = run_shor_factoring(&eng, argv[2]);
        engine_destroy(&eng);
        return ret;
    }

    if (argc >= 2) {
        /* Load and execute program */
        if (load_program(&eng, argv[1]) != 0) {
            engine_destroy(&eng);
            return 1;
        }
        execute_program(&eng);
    } else {
        /* Interactive mode */
        interactive_loop(&eng);
    }

    engine_destroy(&eng);
    return 0;
}
