set pagination off
set breakpoint pending on
set print thread-events off
set stop-on-solib-events 0

# keep noise down for common runtime signals.
handle SIGPIPE nostop noprint pass

# set up a log file
set logging file gdb-mirage-stack-smash.log
set logging overwrite on
set logging enabled on

# likely fatal path for stack smash fail
break __stack_chk_fail
catch signal SIGABRT

# mirage-specific probes (pending breakpoints are fine before .so load).
# we expect later loading from zg
break mirage_graph_superoptimize
break mirage::search::KernelGraphGenerator::show_statistics
rbreak mirage_graph_superoptimize
rbreak .*KernelGraphGenerator::show_statistics.*

define probe_symbols
  echo \n===== probe_symbols =====\n
  info functions mirage_graph_superoptimize
  info functions KernelGraphGenerator::show_statistics
  echo =========================\n
end

document probe_symbols
Search loaded symbols for likely Mirage function names.
Run this after the process has started and shared libs are loaded.
end

define dump_crash
  echo \n===== dump_crash =====\n
  bt full
  echo \n----- all threads -----\n
  thread apply all bt full
  echo \n----- shared libs -----\n
  info sharedlibrary
  echo \n======================\n
end

document dump_crash
Dump useful crash context:
  - current thread full backtrace
  - all threads full backtraces
  - loaded shared libraries
end

echo Loaded gdb-mirage-stack-smash.gdb\n
echo Next: run\n
echo On crash: dump_crash\n
