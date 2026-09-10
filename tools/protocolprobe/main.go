// Generate genuine duplicate-vote evidence for a disposable local test validator.
// This intentionally equivocates. The lab runner supplies only its own generated key.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cometbft/cometbft/privval"
	rpchttp "github.com/cometbft/cometbft/rpc/client/http"
	"github.com/cometbft/cometbft/types"
	tenderminttypes "github.com/cometbft/cometbft/proto/tendermint/types"
)

func main() {
	home := flag.String("home", "", "Disposable validator home")
	rpcURL := flag.String("rpc", "", "Disposable chain RPC")
	height := flag.Int64("height", 0, "Past height at which this validator had voting power")
	flag.Parse()
	if *home == "" || *rpcURL == "" || *height < 1 { panic("home, rpc, and height are required") }
	client, err := rpchttp.New(*rpcURL, "/websocket")
	check(err)
	ctx := context.Background()
	block, err := client.Block(ctx, height)
	check(err)
	page, count := 1, 100
	vs, err := client.Validators(ctx, height, &page, &count)
	check(err)
	if vs.Total != len(vs.Validators) { panic("test supports at most 100 validators") }
	set := types.NewValidatorSet(vs.Validators)
	pv := privval.LoadFilePV(filepath.Join(*home, "config/priv_validator_key.json"), filepath.Join(*home, "data/priv_validator_state.json"))
	index, validator := set.GetByAddress(pv.Key.Address)
	if validator == nil { panic("key was not a validator at the selected height") }
	a := &types.Vote{Type: tenderminttypes.PrecommitType, Height: *height, Round: 0,
		BlockID: block.BlockID, Timestamp: block.Block.Time,
		ValidatorAddress: pv.Key.Address, ValidatorIndex: index}
	b := a.Copy()
	hash := sha256.Sum256([]byte("neuroshard/disposable-equivocation-test"))
	b.BlockID.Hash = hash[:]
	for _, vote := range []*types.Vote{a, b} {
		signature, err := pv.Key.PrivKey.Sign(types.VoteSignBytes(block.Block.ChainID, vote.ToProto()))
		check(err)
		vote.Signature = signature
	}
	evidence, err := types.NewDuplicateVoteEvidence(a, b, block.Block.Time, set)
	check(err)
	result, err := client.BroadcastEvidence(ctx, evidence)
	check(err)
	encoded, err := json.Marshal(map[string]interface{}{"evidence_hash": hex.EncodeToString(result.Hash), "offense_height": *height,
		"validator_address": pv.Key.Address.String()})
	check(err)
	fmt.Println(string(encoded))
}

func check(err error) { if err != nil { fmt.Fprintln(os.Stderr, err); os.Exit(1) } }
