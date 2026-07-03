package kv

import "testing"

func TestReserveCommitRelease(t *testing.T) {
	a := NewAllocator(8, 4) // 8 blocks x 4 tokens = 32 tokens capacity
	s := a.NewSequence()

	if err := s.Reserve(5); err != nil { // 5 tokens -> 2 blocks
		t.Fatal(err)
	}
	if len(s.Table) != 2 || a.FreeBlocks() != 6 {
		t.Fatalf("table=%v free=%d, want 2 blocks / 6 free", s.Table, a.FreeBlocks())
	}
	s.Commit(5)

	if err := s.Reserve(3); err != nil { // 8 tokens total -> still 2 blocks
		t.Fatal(err)
	}
	if len(s.Table) != 2 {
		t.Fatalf("table grew to %v for tokens that fit", s.Table)
	}
	s.Commit(3)

	if err := s.Reserve(1); err != nil { // 9th token -> 3rd block
		t.Fatal(err)
	}
	if len(s.Table) != 3 || a.FreeBlocks() != 5 {
		t.Fatalf("table=%v free=%d, want 3 blocks / 5 free", s.Table, a.FreeBlocks())
	}

	s.Release()
	if a.FreeBlocks() != 8 || s.Len != 0 || s.Table != nil {
		t.Fatalf("release did not return blocks: free=%d len=%d table=%v",
			a.FreeBlocks(), s.Len, s.Table)
	}
}

func TestOutOfBlocks(t *testing.T) {
	a := NewAllocator(2, 4)
	s1, s2 := a.NewSequence(), a.NewSequence()
	if err := s1.Reserve(8); err != nil { // takes both blocks
		t.Fatal(err)
	}
	if err := s2.Reserve(1); err == nil {
		t.Fatal("expected out-of-blocks error")
	}
	s1.Release()
	if err := s2.Reserve(1); err != nil { // blocks recycled
		t.Fatal(err)
	}
}

func TestTablesAreDistinct(t *testing.T) {
	a := NewAllocator(4, 2)
	s1, s2 := a.NewSequence(), a.NewSequence()
	if err := s1.Reserve(4); err != nil {
		t.Fatal(err)
	}
	if err := s2.Reserve(4); err != nil {
		t.Fatal(err)
	}
	seen := map[int32]bool{}
	for _, id := range append(append([]int32{}, s1.Table...), s2.Table...) {
		if seen[id] {
			t.Fatalf("block %d handed out twice", id)
		}
		seen[id] = true
	}
}
