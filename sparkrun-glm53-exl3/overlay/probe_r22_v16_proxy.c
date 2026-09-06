// Execute the actual proxy's CQ accounting without an HCA. Built on ARM64.
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <string.h>

static int poll_calls, poll_result;
static enum ibv_wc_status completion_status;
static int test_poll_cq(struct ibv_cq *cq, int count, struct ibv_wc *wc) {
    (void)cq;
    assert(count == 32);
    poll_calls++;
    if (poll_result > 0) {
        memset(wc, 0, sizeof(*wc));
        wc[0].wr_id = 1;
        wc[0].status = completion_status;
    }
    return poll_result;
}
#define ibv_poll_cq test_poll_cq
#include "_roce_proxy.c"
#undef ibv_poll_cq

int main(void) {
    roce_ctx_t c;
    memset(&c, 0, sizeof(c));
    c.skip_empty_cq = 1;
    assert(drain_cq(&c, 0) == 0 && poll_calls == 0);
    c.hca[0].pending_completions = 1;
    c.hca[0].outstanding[1] = 1;
    poll_result = 0;
    assert(drain_cq(&c, 0) == 0 && poll_calls == 1);
    assert(c.hca[0].pending_completions == 1);
    poll_result = 1;
    completion_status = IBV_WC_SUCCESS;
    assert(drain_cq(&c, 0) == 0 && poll_calls == 2);
    assert(c.hca[0].pending_completions == 0 && c.hca[0].outstanding[1] == 0);
    assert(c.writes_completed == 1 && c.hca[0].writes_completed == 1);
    assert(drain_cq(&c, 0) == 0 && poll_calls == 2);
    c.skip_empty_cq = 0;
    poll_result = 0;
    assert(drain_cq(&c, 0) == 0 && poll_calls == 3);
    c.skip_empty_cq = 1;
    c.hca[0].pending_completions = c.hca[0].outstanding[1] = 1;
    poll_result = 1;
    completion_status = IBV_WC_LOC_PROT_ERR;
    assert(drain_cq(&c, 0) == -1 && poll_calls == 4);
    assert(c.hca[0].pending_completions == 1 && c.hca[0].outstanding[1] == 1);
    assert(strstr(c.err, "failed") != NULL);
    poll_result = -1;
    errno = EIO;
    assert(drain_cq(&c, 0) == -1 && poll_calls == 5);
    puts("v16 native CQ pending/empty/error checks passed");
    return 0;
}
