// loadgen drives the HTTP/SSE server with concurrent mixed-length requests
// and reports throughput + latency percentiles per concurrency level.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	url := flag.String("url", "http://127.0.0.1:8080", "server base URL")
	levels := flag.String("concurrency", "1,2,4,8,16", "comma-separated concurrency levels")
	requests := flag.Int("requests", 32, "requests per level")
	minPrompt := flag.Int("min-prompt", 4, "min prompt length")
	maxPrompt := flag.Int("max-prompt", 48, "max prompt length")
	steps := flag.Int("steps", 32, "max_new_tokens per request")
	vocab := flag.Int("vocab", 512, "vocab size for random prompts")
	flag.Parse()

	fmt.Printf("%-6s %-10s %-12s %-12s %-12s %-10s\n",
		"conc", "req/s", "tok/s", "TTFT p50", "TTFT p99", "total p99")
	for _, ls := range strings.Split(*levels, ",") {
		conc, err := strconv.Atoi(strings.TrimSpace(ls))
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad concurrency level:", ls)
			os.Exit(1)
		}
		if err := runLevel(*url, conc, *requests, *minPrompt, *maxPrompt, *steps, *vocab); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
	}
}

type result struct {
	ttft   time.Duration
	total  time.Duration
	tokens int
	err    error
}

func runLevel(url string, conc, requests, minP, maxP, steps, vocab int) error {
	jobs := make(chan int)
	results := make([]result, requests)
	var wg sync.WaitGroup

	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(worker)*7919 + 17))
			for i := range jobs {
				results[i] = one(url, rng, minP, maxP, steps, vocab)
			}
		}(w)
	}
	start := time.Now()
	for i := 0; i < requests; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	wall := time.Since(start)

	var ttfts, totals []time.Duration
	totalTokens := 0
	for _, r := range results {
		if r.err != nil {
			return r.err
		}
		ttfts = append(ttfts, r.ttft)
		totals = append(totals, r.total)
		totalTokens += r.tokens
	}
	sort.Slice(ttfts, func(i, j int) bool { return ttfts[i] < ttfts[j] })
	sort.Slice(totals, func(i, j int) bool { return totals[i] < totals[j] })
	p := func(ds []time.Duration, q float64) time.Duration { return ds[int(float64(len(ds)-1)*q)] }

	fmt.Printf("%-6d %-10.1f %-12.1f %-12v %-12v %-10v\n",
		conc,
		float64(requests)/wall.Seconds(),
		float64(totalTokens)/wall.Seconds(),
		p(ttfts, 0.5), p(ttfts, 0.99), p(totals, 0.99))
	return nil
}

func one(url string, rng *rand.Rand, minP, maxP, steps, vocab int) result {
	n := minP + rng.Intn(maxP-minP+1)
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(rng.Intn(vocab))
	}
	body, _ := json.Marshal(map[string]any{"ids": ids, "max_new_tokens": steps})

	start := time.Now()
	resp, err := http.Post(url+"/v1/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return result{err: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return result{err: fmt.Errorf("status %s", resp.Status)}
	}

	var r result
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		if strings.Contains(payload, "\"error\"") {
			return result{err: fmt.Errorf("server error: %s", payload)}
		}
		if r.tokens == 0 {
			r.ttft = time.Since(start)
		}
		r.tokens++
	}
	if err := sc.Err(); err != nil {
		return result{err: err}
	}
	r.total = time.Since(start)
	return r
}
