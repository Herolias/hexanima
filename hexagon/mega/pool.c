#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include "pool.h"
#define MAXW 8   /* V69: at most 3 + main (4 HVX contexts); V81 has 12 hardware threads */
static struct { volatile int gen, done, quit; pool_fn fn; void *arg; int n; pthread_t th[MAXW]; int idx[MAXW]; } P;
static void *wk(void *p) {
  int i = *(int *)p, seen = 0;
  for (;;) {
    while (P.gen == seen && !P.quit) {}
    if (P.quit) return 0;
    seen = P.gen; __sync_synchronize();
    P.fn(P.arg, i, P.n);
    __sync_fetch_and_add(&P.done, 1);
  }
}
int pool_init(int n) {
  if (P.n) return 0;
  P.n = n > MAXW ? MAXW : n; P.quit = 0; P.gen = 0; P.done = P.n;
  for (int i = 0; i < P.n; i++) {
    pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 256 * 1024);
    P.idx[i] = i; pthread_create(&P.th[i], &at, wk, &P.idx[i]);
  }
  return 0;
}
void pool_start(pool_fn fn, void *arg) { P.fn = fn; P.arg = arg; P.done = 0; __sync_synchronize(); P.gen++; }
void pool_wait(void) { while (P.done < P.n) {} __sync_synchronize(); }
void pool_run(pool_fn fn, void *arg) { pool_start(fn, arg); pool_wait(); }
void pool_quit(void) { if (!P.n) return; P.quit = 1; for (int i = 0; i < P.n; i++) pthread_join(P.th[i], 0); P.n = 0; }
int pool_n(void) { return P.n; }
