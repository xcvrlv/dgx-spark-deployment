// Execute the actual proxy's CQ accounting without an HCA. Built on ARM64.
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <string.h>

static int poll_calls, poll_result;
static int post_count, payload_inline, payload_bytes;
static int posted_peers[8];
static int test_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                          struct ibv_send_wr **bad) {
    (void)qp; (void)bad;
    assert(post_count < 8);
    posted_peers[post_count++] = (int)wr->wr_id;
    struct ibv_send_wr *flag = wr;
    if (wr->next != NULL) {
        assert(wr->sg_list->length == (unsigned)payload_bytes);
        assert(!!(wr->send_flags & IBV_SEND_INLINE) == payload_inline);
        assert(!(wr->send_flags & IBV_SEND_SIGNALED));
        flag = wr->next;
    } else {
        assert(payload_bytes == 0);
    }
    assert(flag->next == NULL && flag->sg_list->length == 4);
    assert(flag->send_flags == (IBV_SEND_SIGNALED | IBV_SEND_INLINE));
    return 0;
}
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
#define ibv_post_send test_post_send
#include "_roce_proxy.c"
#undef ibv_poll_cq
#undef ibv_post_send

static void test_posting(int rotated, int use_inline, int bytes, int limit, int expected_inline) {
    roce_ctx_t c;
    struct ibv_mr mr;
    uint8_t region[8192];
    memset(&c, 0, sizeof(c));
    memset(&mr, 0, sizeof(mr));
    c.world = 4; c.rank = 1; c.n_hca = 2;
    c.slot_bytes = 4096; c.region = region;
    c.balanced_fanout = rotated; c.inline_payload = use_inline;
    c.skip_empty_cq = 1;
    for (int h = 0; h < 2; ++h) {
        c.hca[h].mr = &mr;
        for (int p = 0; p < 4; ++p) c.hca[h].inline_bytes[p] = limit;
    }
    post_count = 0; poll_result = 0;
    payload_bytes = bytes / 2; payload_inline = expected_inline;
    assert(post_op(&c, 1, bytes) == 0);
    assert(post_count == 6);
    const int normal[] = {0, 0, 2, 2, 3, 3};
    const int balanced[] = {2, 2, 3, 3, 0, 0};
    for (int i = 0; i < 6; ++i) assert(posted_peers[i] == (rotated ? balanced[i] : normal[i]));
    for (int h = 0; h < 2; ++h) {
        assert(c.hca[h].pending_completions == 3);
        assert(c.hca[h].bytes_posted == (unsigned)(3 * bytes / 2));
        assert(c.hca[h].outstanding[1] == 0);
    }
}

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
    test_posting(0, 0, 128, 64, 0);
    test_posting(1, 1, 128, 64, 1);
    test_posting(1, 1, 256, 64, 0);
    test_posting(1, 1, 128, 16, 0);
    test_posting(1, 1, 32, 16, 1);
    puts("RoCEnante CQ/error, fanout, inline-limit and payload-before-flag checks passed");
    return 0;
}
