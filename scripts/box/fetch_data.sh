#!/usr/bin/env bash
# Download the training-mix datasets from the public HF repo, extract them and rebuild the mixes (run in ~/pi).
set -eu
BASE=https://huggingface.co/datasets/KuanKuanKuan/so101-pi07-sim-data/resolve/main
mkdir -p ~/hf_data && cd ~/hf_data
FILES="sim_v1/sim_v1.tar.part-000 sim_ladder_v1/sim_ladder_v1.tar.part-000 sim_ladder_v1/sim_ladder_v1.tar.part-001
sim_ladder_v1/sim_ladder_v1.tar.part-002 sim_recover2_v1/sim_recover2_v1.tar.part-000 sim_recover2_v1/sim_recover2_v1.tar.part-001
sim_recover2_v1/sim_recover2_v1.tar.part-002 sim_recover2_v1/sim_recover2_v1.tar.part-003 sim_recover2_v1/sim_recover2_v1.tar.part-004
sim_twobins_mixed_v1/sim_twobins_mixed_v1.tar.part-000 sim_twobins_mixed_v1/sim_twobins_mixed_v1.tar.part-001
mixes/sim_mix_v1.txt mixes/sim_mixdagclean_v1.txt sim_mix_v1_pi07_meta/stats.json sim_mix_v1_pi07_meta/fast/fast.json
sim_mix_v1_pi07_meta/fast/fast_bpe.json"
for v in 1 2 3 4 5 6 7 9; do FILES="$FILES sim_dagger_v$v/sim_dagger_v$v.tar.part-000"; done
echo $FILES | tr ' ' '\n' | xargs -P 8 -I{} sh -c 'mkdir -p $(dirname {}) && curl -sSfL --retry 5 -o {} "'$BASE'/{}" && echo got {}'
cd ~/pi/data
for d in sim_v1 sim_ladder_v1 sim_recover2_v1 sim_twobins_mixed_v1 sim_dagger_v1 sim_dagger_v2 sim_dagger_v3 sim_dagger_v4 sim_dagger_v5 sim_dagger_v6 sim_dagger_v7 sim_dagger_v9; do
  [ -d $d ] || (cat ~/hf_data/$d/$d.tar.part-* | tar xf - && echo extracted $d) &
done
wait
for mix in sim_mix_v1 sim_mixdagclean_v1; do
  mkdir -p $mix
  (cd $mix && while read -r name arrow target; do ln -sfn "$target" "$name"; done < ~/hf_data/mixes/$mix.txt)
  echo "mix $mix: $(ls $mix | wc -l) links"
done
mkdir -p sim_mix_v1/pi07_meta/fast && cp ~/hf_data/sim_mix_v1_pi07_meta/stats.json sim_mix_v1/pi07_meta/ && cp ~/hf_data/sim_mix_v1_pi07_meta/fast/* sim_mix_v1/pi07_meta/fast/
echo DATA READY
