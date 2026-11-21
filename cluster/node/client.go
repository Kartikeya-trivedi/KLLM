package node

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"kllm/cluster"
)

// Client is how the orchestrator reaches a node. *Node satisfies it for
// in-process clusters (tests, fast dataset generation); HTTPClient reaches
// one over the network.
type Client interface {
	ID() string
	Infer(ctx context.Context, req cluster.Request) cluster.Response
	Stats(ctx context.Context) (Stats, error)
}

// HTTPClient talks to a node's HTTP surface.
type HTTPClient struct {
	id, base string
	hc       *http.Client
}

// NewHTTPClient addresses the node at baseURL. id is the name the caller
// knows the node by; responses are stamped with it regardless of what the
// node claims, so a node cannot answer as somebody else.
func NewHTTPClient(id, baseURL string) *HTTPClient {
	return &HTTPClient{
		id:   id,
		base: strings.TrimRight(baseURL, "/"),
		hc: &http.Client{Transport: &http.Transport{
			MaxIdleConnsPerHost: 64,
			IdleConnTimeout:     90 * time.Second,
		}},
	}
}

func (c *HTTPClient) ID() string  { return c.id }
func (c *HTTPClient) URL() string { return c.base }

func (c *HTTPClient) Infer(ctx context.Context, req cluster.Request) cluster.Response {
	start := time.Now()
	resp := c.infer(ctx, req)
	resp.RequestID, resp.Node = req.ID, c.id
	if len(resp.Conf) != len(resp.Tokens) {
		resp.Conf = nil // malformed self-report: treat as unreported
	}
	// A realtime node's own figure excludes the network; a virtual-time
	// node's figure is the only one that means anything. The larger of
	// the two is right in both cases.
	if observed := float64(time.Since(start)) / float64(time.Millisecond); observed > resp.LatencyMs {
		resp.LatencyMs = observed
	}
	return resp
}

func (c *HTTPClient) infer(ctx context.Context, req cluster.Request) cluster.Response {
	body, _ := json.Marshal(req)
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/v1/infer", bytes.NewReader(body))
	if err != nil {
		return cluster.Response{Err: err.Error()}
	}
	hreq.Header.Set("Content-Type", "application/json")
	hres, err := c.hc.Do(hreq)
	if err != nil {
		if ctx.Err() != nil {
			err = ctx.Err()
		}
		return cluster.Response{Err: err.Error()}
	}
	defer hres.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(hres.Body, 16<<20))
	if err != nil {
		return cluster.Response{Err: err.Error()}
	}
	var resp cluster.Response
	if json.Unmarshal(raw, &resp) != nil {
		resp = cluster.Response{}
	}
	if hres.StatusCode != http.StatusOK && resp.Err == "" {
		resp.Err = fmt.Sprintf("node returned %s: %s", hres.Status, strings.TrimSpace(string(raw)))
	}
	if resp.Err != "" {
		resp.Tokens, resp.Conf = nil, nil
	}
	return resp
}

func (c *HTTPClient) Stats(ctx context.Context) (Stats, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/stats.json", nil)
	if err != nil {
		return Stats{}, err
	}
	hres, err := c.hc.Do(hreq)
	if err != nil {
		return Stats{}, err
	}
	defer hres.Body.Close()
	if hres.StatusCode != http.StatusOK {
		return Stats{}, fmt.Errorf("node %s: stats returned %s", c.id, hres.Status)
	}
	var s Stats
	if err := json.NewDecoder(hres.Body).Decode(&s); err != nil {
		return Stats{}, err
	}
	s.Node = c.id
	return s, nil
}

// Remote is a Backend that forwards to another node-protocol server. Point
// it at `kllm serve` and the real engine can run behind a fault profile.
type Remote struct {
	C Client
}

func (r Remote) Model() string { return "remote:" + r.C.ID() }

func (r Remote) Generate(ctx context.Context, req cluster.Request) (Output, error) {
	resp := r.C.Infer(ctx, req)
	if !resp.OK() {
		return Output{}, errors.New(resp.Err)
	}
	out := Output{Tokens: resp.Tokens, Conf: resp.Conf, Label: cluster.Label{Fault: FaultNone}}
	if resp.Label != nil {
		out.Label = *resp.Label
	}
	return out, nil // Latency stays zero: the upstream really spent the time
}
