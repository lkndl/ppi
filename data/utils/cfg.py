import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Set

from dataclass_wizard import JSONWizard


class SamplingStrategy(Enum):
    RANDOM = 0
    BALANCED = 1


class CorrelationType(Enum):
    PEARSON = 0
    SPEARMAN = 1


@dataclass
class Config(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = 'SNAKE'

    keep_human: bool = False
    keep_interspecies: bool = False
    accept_homodimers: bool = False
    add_proteomes: bool = True
    slurm: bool = all(shutil.which(t) for t in ('snakemake', 'sbatch'))
    slurm_node_limit: int = 15
    rostclust: bool = shutil.which('rostclust')

    cwd: Path = Path.cwd()
    ap: Path = Path('apid')
    hp: Path = Path('huri')
    qp: Path = None
    pp: Path = Path('swissprot')

    min_seq_len: int = 50
    max_seq_len: int = 1500

    psi_path = Path('hi_union.psi')
    hval_config: dict = field(default_factory=dict)
    hval_config_path: Path = Path('hval_config.json')

    seed: int = 42
    ratio: float = 10.0
    strategy: SamplingStrategy = SamplingStrategy.BALANCED

    val_set_size: float = .1
    val_species: Set[str] = field(default_factory=set)
    train_species: Set[str] = field(default_factory=set)

    val_raw_fasta: Path = ap / 'apid_validation_raw.fasta'
    val_raw_tsv: Path = ap / 'apid_validation_raw.tsv'
    val_rr_fasta: Path = ap / 'apid_validation_rr.fasta'
    val_rr_tsv: Path = ap / 'apid_validation_rr.tsv'
    val_sp_fasta: Path = ap / 'apid_validation_sp.fasta'
    val_sp_tsv: Path = ap / 'apid_validation_sp.tsv'
    val_c3_fasta: Path = ap / 'apid_validation_c3.fasta'
    val_fasta: Path = Path('apid_validation.fasta')
    val_tsv: Path = Path('apid_validation.tsv')

    test_raw_fasta: Path = hp / 'huri_test_raw.fasta'
    test_raw_tsv: Path = hp / 'huri_test_raw.tsv'
    test_rr_fasta: Path = hp / 'huri_test_rr.fasta'
    test_rr_tsv: Path = hp / 'huri_test_rr.tsv'
    test_sp_fasta: Path = hp / 'huri_test_sp.fasta'
    test_sp_tsv: Path = hp / 'huri_test_sp.tsv'
    test_c3_fasta: Path = hp / 'huri_test_c3.fasta'
    test_fasta: Path = Path('huri_test.fasta')
    test_tsv: Path = Path('huri_test.tsv')

    train_tsv: Path = Path('apid_train.tsv')
    train_fasta: Path = Path('apid_train.fasta')
    train_raw_fasta: Path = ap / 'apid_train_raw.fasta'
    train_raw_tsv: Path = ap / 'apid_train_raw.tsv'

    train_proteome: Path = Path('train_proteome.json')
    val_proteome: Path = Path('val_proteome.json')
    test_proteome: Path = Path('test_proteome.json')

    # results
    weird_species: set[int] = field(default_factory=set)
    train_bias: str = None
    train_seqs: int = None
    train_extra: int = None
    train_size: int = None
    test_seqs: int = None
    test_extra: int = None
    val_seqs: int = None
    val_extra: int = None
    val_bias: str = None
    val_size: int = None
    val_sizes: str = None
    test_bias: float = None
    test_size: int = None
    test_sizes: str = None
