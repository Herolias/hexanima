#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
int main(int argc, char **argv) {
  int K = argc > 1 ? atoi(argv[1]) : 256, reps = argc > 2 ? atoi(argv[2]) : 1000, stage = argc > 3 ? atoi(argv[3]) : 2;
  struct remote_rpc_control_unsigned_module d = {CDSP_DOMAIN_ID, 1};
  int r = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &d, sizeof d);
  printf("unsigned ctl rc=%d\n", r);
  remote_handle64 h = 0;
  r = mega_open(mega_URI CDSP_DOMAIN, &h);
  printf("open rc=0x%x\n", r); if (r) return 1;
  { int ar, hw, hv, vt, va, cp, cv; int ri = mega_info(h, &ar, &hw, &hv, &vt, &va, &cp, &cv);
    printf("info rc=0x%x arch=0x%x hwthreads=%d hvx=0x%x vtcm=%d KB (avail %d KB) caps=0x%x hmx_store=%s\n", ri, ar, hw, hv, vt / 1024, va / 1024, cp, cv ? "v73+" : "v69"); }
  long long cyc; float me; int lk, vt;
  struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
  r = mega_probe(h, K, reps, stage, &cyc, &me, &lk, &vt);
  clock_gettime(CLOCK_MONOTONIC, &b);
  double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
  printf("probe rc=0x%x vtcm=%d hmxlock=%d maxerr=%g cyc/call=%lld  wall=%.2fms\n", r, vt, lk, me, cyc, ms);
  mega_close(h);
}
