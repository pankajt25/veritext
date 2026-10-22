"""Evaluation dataset builder generating 110 benchmark documents across groups a-h and manifest.csv."""

import csv
import json
from pathlib import Path
import random

from veritext.report.builder import build_report
from veritext.report.seal import compute_seal

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DIR = BASE_DIR / "data" / "eval"
REFS_DIR = EVAL_DIR / "references"
DOCS_DIR = EVAL_DIR / "documents"


def build_evaluation_dataset() -> Path:
    """Constructs the complete synthetic evaluation dataset of 110 documents and manifest.csv."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Read references
    ref_files = sorted(list(REFS_DIR.glob("ref_*.txt")))
    if not ref_files:
        raise FileNotFoundError(f"Reference files missing in {REFS_DIR}. Run populate_refs first.")

    refs = {f.name: f.read_text(encoding="utf-8") for f in ref_files}

    manifest_rows = []

    # --------------------------------------------------------------------------
    # GROUP A: 15 Verbatim or Lightly Edited Copies (with Obfuscations)
    # --------------------------------------------------------------------------
    homoglyphs = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}  # Cyrillic lookalikes

    for i in range(1, 16):
        fname = f"copy_{i:02d}.txt"
        ref_key = f"ref_{((i - 1) % 10) + 1:02d}.txt"
        base_text = refs[ref_key]

        if i <= 8:
            # Exact verbatim copy
            content = base_text
            note = f"Verbatim copy of {ref_key}"
        elif i <= 10:
            # Lightly edited copy: replace 3 words with synonyms
            content = base_text.replace("Modern", "Contemporary").replace("Traditional", "Standard").replace("mechanism", "system")
            note = f"Lightly edited copy of {ref_key}"
        elif i <= 13:
            # Homoglyph obfuscation
            content_chars = list(base_text)
            obf_count = 0
            for idx_c, ch in enumerate(content_chars):
                if ch in homoglyphs and obf_count < 15:
                    content_chars[idx_c] = homoglyphs[ch]
                    obf_count += 1
            content = "".join(content_chars)
            note = f"Homoglyph obfuscated copy of {ref_key}"
        else:
            # Zero-width space obfuscation (\u200b)
            words = base_text.split()
            obf_words = []
            for w_idx, w in enumerate(words):
                if w_idx % 4 == 0 and len(w) > 3:
                    w = w[:2] + "\u200b" + w[2:]
                obf_words.append(w)
            content = " ".join(obf_words)
            note = f"Zero-width space obfuscated copy of {ref_key}"

        (DOCS_DIR / fname).write_text(content, encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "a_copies",
            "expected_plagiarism": "yes",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": note,
        })

    # --------------------------------------------------------------------------
    # GROUP B: 15 Paraphrased Versions of Reference Passages
    # --------------------------------------------------------------------------
    PARAPHRASES = [
        # Para 01 (derived from Ref 01: Attention Mechanisms)
        (
            "Contemporary neural network systems rely heavily upon self-attention concepts when handling sequential data. "
            "Prior recurrent architectures analyzed inputs in a step-by-step fashion, which frequently restricted the retention "
            "of distant relational information across sentences. Multi-head self-attention circumvents this bottleneck by computing "
            "simultaneous similarity weights between all elements in a sequence. By projecting key, query, and value tensors into "
            "separate spatial subspaces, the model can capture multifaceted semantic relationships from diverse contextual viewpoints. "
            "Additionally, position-aware vectors are incorporated into the token inputs to preserve structural sequence geometry. "
            "This mathematical design affords significant parallel execution during the optimization phase, shortening compute "
            "time while yielding noticeably superior representations across various natural language translation and comprehension benchmarks."
        ),
        # Para 02 (derived from Ref 02: Locality-Sensitive Hashing)
        (
            "Locality-sensitive hashing offers a practical approach for discovering approximate nearest neighbors inside high-dimensional spaces. "
            "Whereas typical hash algorithms aim to avoid collisions by scattering values evenly across hash buckets, locality-sensitive hashing "
            "deliberately promotes collisions for vectors that share high geometric proximity. For set overlap analysis, MinHash vectors produce "
            "an unbiased approximation of the underlying Jaccard coefficient between token sets. Partitioning the MinHash array into multiple bands "
            "and storing each band inside separate index tables enables candidate pairs to be identified with sub-quadratic computational overhead. "
            "Pairs colliding across any individual band are then flagged for thorough pairwise comparison, allowing high-throughput similarity "
            "screening on ordinary desktop computers."
        ),
        # Para 03 (derived from Ref 03: Consensus Protocols)
        (
            "Consensus mechanisms in decentralized environments ensure that clustered nodes maintain an identical replicated state, even "
            "when facing node outages or communication disruptions. Although classical protocols like Paxos ensure safety during asynchronous "
            "network conditions, their intricate operational principles often make practical implementation challenging. In contrast, the Raft "
            "protocol was structured specifically for conceptual clarity by partitioning the overall problem into distinct components: electing "
            "a leader, propagating state logs, and guaranteeing consistency. Server nodes alternate between follower, candidate, and leader roles "
            "governed by timed heartbeats. The elected quorum leader coordinates state modifications and records entries permanently onto disk, "
            "guaranteeing that confirmed transactions persist immutably throughout the operational network."
        ),
        # Para 04 (derived from Ref 04: Photosynthetic Carbon Fixation)
        (
            "Photosynthetic assimilation represents the biological sequence by which autotrophs convert ambient carbon dioxide into nutritious sugars. "
            "The dark Calvin-Benson reaction starts when the RuBisCO enzyme adds carbon to a five-carbon acceptor molecule. The resulting unstable "
            "compound splits into twin molecules of 3-phosphoglycerate, which are subsequently reduced using ATP and NADPH generated during "
            "sunlight absorption. These triose molecules are then harvested for carbohydrate production or used to regenerate the initial sugar "
            "acceptor. In hotter environments, competing oxygenation reactions cause photorespiratory energy losses, prompting evolutionary adaptations "
            "such as C4 and CAM pathways that isolate carbon concentration inside specialized plant compartments."
        ),
        # Para 05 (derived from Ref 05: Zero-Knowledge Proofs)
        (
            "Zero-knowledge cryptography enables a proving party to convince an inspector that a specific mathematical claim holds true without "
            "revealing any private underlying data. Founded upon the mathematical criteria of completeness, soundness, and zero-knowledge, these "
            "interactive protocols transform computational logic into algebraic circuits. Non-interactive succinct arguments leverage structured "
            "reference strings and elliptic pairings to compress assertions into compact proofs that can be authenticated rapidly by any observer. "
            "Key implementations encompass confidential identity validation, private transactions on distributed ledgers, and delegating intensive "
            "computations from low-powered hardware to remote cloud servers without compromising secrecy."
        ),
        # Para 06 (derived from Ref 06: Subduction Zones)
        (
            "Subduction boundaries are geological collision zones where heavy oceanic plates sink beneath lighter continental masses into the mantle. "
            "As this water-rich oceanic crust descends into regions of increasing heat and pressure, dehydration metamorphic changes release trapped "
            "moisture into the surrounding mantle wedge. This fluid influx dramatically lowers the melting point of neighboring peridotite rocks, "
            "producing buoyant basaltic and andesitic magma. The rising molten material fuels active volcanic chains and aids in the formation "
            "of continental landmasses. Meanwhile, locked fault surfaces along the plate contact store massive tectonic tension, occasionally "
            "unleashing severe megathrust earthquakes capable of triggering devastating oceanwide tsunamis."
        ),
        # Para 07 (derived from Ref 07: Quantum Decoherence)
        (
            "The quantum measurement dilemma involves how reversible unitary wave behavior transforms into distinct classical observational data. "
            "Decoherence theory explains this transition through continuous entanglement with surrounding environments, wherein microscopic quantum "
            "states continuously interact with ambient particles and thermal energy. These pervasive disturbances disperse quantum phase relationships "
            "into macroscopic environmental degrees of freedom that cannot be observed. As a consequence, the system's reduced density matrix rapidly "
            "diagonalizes along a preferred pointer basis governed by the interaction Hamiltonian. While decoherence clarifies why macroscopic superpositions "
            "fade from view, it preserves overall unitary evolution, leaving individual outcome selection subject to theoretical debate."
        ),
        # Para 08 (derived from Ref 08: Epigenetics)
        (
            "Epigenetic mechanisms encompass transmissible variations in cellular expression and phenotypic traits that occur without altering the "
            "underlying DNA sequence. Structural chromatin remodeling, including DNA methylation and chemical histone modifications, determines whether "
            "specific genomic regions remain accessible to RNA polymerase. Adding methyl groups to promoter CpG islands commonly suppresses transcription "
            "by recruiting binding proteins that condense chromatin into dense heterochromatin. Conversely, histone acetyltransferases attach acetyl "
            "groups to lysine residues, reducing positive charges and loosening DNA-histone bonds. These dynamic molecular modifications maintain stable "
            "cell identity while remaining adaptable to metabolic cues."
        ),
        # Para 09 (derived from Ref 09: Behavioral Economics)
        (
            "Standard economic models assume that human actors behave as rational utility maximizers possessing unlimited reasoning abilities and "
            "fixed goals. Behavioral economics refutes this assumption by highlighting consistent cognitive shortcuts that depart from rational rules. "
            "Prospect theory indicates that people assess outcomes as profits or losses measured against a subjective baseline rather than overall wealth. "
            "Furthermore, empirical curves demonstrate an S-shape with distinct loss aversion, meaning the emotional pain of losing exceeds the satisfaction "
            "of gaining an equal amount. Other bounded rationality traits, including present-bias discounting and contextual framing, significantly "
            "impact market prices and household savings behavior."
        ),
        # Para 10 (derived from Ref 10: Bacterial Biofilms)
        (
            "Bacterial biofilms are dense colonies of microorganisms embedded in a protective matrix of extracellular polymers, including proteins, "
            "sugars, and free DNA. This protective architecture establishes steep gradients in nutrient availability, pH, and oxygen, nurturing specialized "
            "dormant persister cells. In medical environments, biofilms adhere strongly to artificial implants and biological membranes, exhibiting "
            "resistance against antibiotics that can be hundreds of times higher than that of free-swimming cells. Physical diffusion limitations hinder "
            "drug entry, while matrix enzymes and elevated gene transfer accelerate the spread of multi-drug resistance across the bacterial population."
        ),
        # Para 11 (derived from Ref 11: Graph Neural Networks)
        (
            "Graph neural networks extend deep learning principles to non-grid relational structures where information is mapped as vertices and edges. "
            "The core computing paradigm relies on iterative message passing, in which every node synthesizes features from adjacent neighboring nodes "
            "over successive hidden layers. An updating step then integrates these aggregated inputs with the node's prior embedding. Permutation invariance "
            "ensures that graph isomorphisms yield identical outputs. Nonetheless, deeper graph architectures often suffer from over-smoothing, "
            "wherein embeddings of disparate nodes blend into uniform vectors, restricting useful model depth in molecular analysis and social networks."
        ),
        # Para 12 (derived from Ref 12: Urban Heat Islands)
        (
            "The urban heat island effect highlights how cities consistently register warmer temperatures than adjacent undeveloped rural zones. "
            "Human heat generation, diminished plant evapotranspiration, and ubiquitous dark construction materials like concrete absorb immense solar "
            "radiation during the day. During nighttime hours, this stored thermal mass releases energy sluggishly, impairing human rest and driving up "
            "air conditioning electricity consumption. Effective remedies center on expanding urban vegetation, including planted roofs, extensive tree "
            "planting, and permeable ground surfaces. High-reflectance coatings reflect sunlight back into the atmosphere, decreasing peak surface heat."
        ),
        # Para 13 (derived from Ref 13: Synaptic Plasticity)
        (
            "Synaptic malleability serves as the cellular foundation for learning and memory storage across the nervous system. Long-term potentiation, "
            "first documented in hippocampal tissue, involves a sustained increase in communication strength across synapses following repetitive "
            "electrical stimulation. This enhancement requires postsynaptic depolarization strong enough to displace magnesium ions that block NMDA "
            "receptor channels. Inflowing calcium ions then trigger CaMKII enzymes, which modify existing AMPA receptors and drive the insertion of new "
            "receptors into the postsynaptic membrane, strengthening subsequent excitatory responses."
        ),
        # Para 14 (derived from Ref 14: Cellular Automata)
        (
            "Cellular automata represent mathematical dynamical models composed of a grid of discrete cells that switch between finite allowable "
            "states. Updates happen synchronously at regular intervals following uniform local rules that consider solely the states of adjacent neighboring "
            "cells. Despite their straightforward design, deterministic cellular automata frequently display complex emergent patterns, organization, "
            "and computational completeness, as seen in Conway's Game of Life. Wolfram's taxonomy splits automata behaviors into four categories, "
            "from static uniformity to chaotic randomness and complex computational forms, demonstrating how rich behaviors emerge from simple foundations."
        ),
        # Para 15 (derived from Ref 15: Hydrothermal Vents)
        (
            "Deep sea hydrothermal vents located along ocean ridges host vibrant chemosynthetic biological systems without requiring sunlight. "
            "Frigid ocean water seeps into fractured volcanic bedrock, where magma sources heat the liquid up to several hundred degrees Celsius while "
            "leaching minerals and sulfur compounds. Upon venting into the freezing ocean depths, these dissolved minerals precipitate quickly to form "
            "towering mineral chimneys known as black smokers. Chemosynthetic bacteria harness energy from oxidising hydrogen sulfide, forming the base "
            "of the food chain that sustains specialized vent creatures such as gigantic tube worms and vent clams."
        ),
    ]

    for i, p_text in enumerate(PARAPHRASES, 1):
        fname = f"para_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(p_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "b_paraphrases",
            "expected_plagiarism": "yes",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"Paraphrase of ref_{i:02d}.txt",
        })

    # --------------------------------------------------------------------------
    # GROUP C: 15 Fully Original Human Passages on Different Topics
    # --------------------------------------------------------------------------
    ORIGINALS = [
        "The transition from two-field to three-field crop rotation in medieval European agriculture substantially increased grain yields and improved soil fertility across the continent. Under the earlier Roman two-field system, half of all arable land remained fallow each year to replenish moisture and nitrogen. The three-field innovation partitioned arable land into three sections: one planted in autumn with winter wheat or rye, a second planted in spring with protein-rich legumes, oats, or barley, and a third left fallow. Legumes harbored symbiotic nitrogen-fixing rhizobia bacteria that actively regenerated depleted soil nutrients. This agricultural schedule reduced famine vulnerability by diversifying harvest timing, provided essential fodder for heavier draught horses, and supported significant demographic growth across northern European feudal manors. Furthermore, the systematic introduction of the heavy wheeled mouldboard plow allowed peasant communities to exploit rich, moist clay soils that had previously resisted cultivation by lighter scratch plows. The resulting agrarian surpluses catalyzed urbanization, market emergence, and demographic stabilization throughout the High Middle Ages across Western Europe.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "Quantum key distribution exploits fundamental physical principles of quantum mechanics rather than computational hardness assumptions to establish shared cryptographic keys between distant parties. The pioneering BB84 protocol encodes random classical bit values into polarized photon states chosen randomly from two non-orthogonal bases, such as rectilinear and diagonal polarizations. According to the quantum no-cloning theorem, an eavesdropper cannot copy unknown quantum states without measurement. Furthermore, quantum measurement inevitably perturbs non-orthogonal states, introducing detectable bit error rates during reconciliation phases. If the observed quantum bit error rate remains below a strict theoretical threshold, privacy amplification extracts an unconditionally secure secret key. Experimental implementations utilize single-photon avalanche detectors and fiber optic links or satellite transceivers to transmit optical pulses across hundreds of kilometers. Recent advancements in decoy-state protocols and measurement-device-independent architectures have substantially mitigated detector side-channel vulnerabilities in commercial metropolitan optical networks.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "Reef-building stony corals maintain an obligatory mutualistic symbiosis with photosynthetic dinoflagellates of the family Symbiodiniaceae. These microscopic endosymbionts reside within the gastrodermal cells of coral polyps, translocating up to ninety percent of their photosynthetically fixed carbon to the animal host in exchange for inorganic nutrients. When seawater temperatures exceed regional summer maximums by as little as one degree Celsius, excessive thermal and irradiance stress damages the photosynthetic apparatus of the dinoflagellates, triggering elevated production of cytotoxic reactive oxygen species. To protect their cellular integrity, coral hosts expel the damaged algal symbionts, revealing the translucent white calcium carbonate skeleton beneath in a process termed coral bleaching. Prolonged bleaching events deplete metabolic glycogen reserves, rendering colonies susceptible to opportunistic microbial pathogens and starvation. Unless sea temperatures return promptly to historical seasonal baselines, widespread coral mortality follows, precipitating ecological regime shifts toward fleshy macroalgal dominance across degraded tropical reef habitats.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "The development of Renaissance polyphony marked a profound transformation in European sacred musical composition during the fifteenth and sixteenth centuries. Composers like Josquin des Prez and Giovanni Pierluigi da Palestrina moved away from the rigid mathematical isorythms of medieval organum toward smooth imitative polyphony, in which each vocal part shared melodic material equally. The humanistic emphasis on rhetorical expression demanded that musical contours mirror the natural accentuation of Latin textual phrasing. Strict counterpoint rules governed the treatment of dissonance, requiring harsh intervals to resolve stepwise onto harmonious imperfect and perfect consonances on unaccented beats, producing an ethereal acoustic texture that exemplified the liturgical ideals of the Council of Trent. Modal voice leading principles ensured clear textual intelligibility even amid complex five-part and six-part vocal textures. This refined polyphonic tradition established compositional foundations for choral harmonic balance and vocal register separation that profoundly influenced succeeding Baroque cantata and fugal writing throughout Western music history.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "CRISPR base editing represents a precision genome editing technology capable of directly converting one target base pair into another without creating double-stranded DNA breaks or requiring donor repair templates. Developed by fusing a catalytically impaired Cas9 nickase with an engineered deaminase enzyme, base editors target specific genetic loci guided by short single-guide RNA sequences. Cytosine base editors utilize cytidine deaminase to chemically convert cytosine into uracil within a five-nucleotide editing window, which DNA replication subsequently resolves into thymine. Adenine base editors similarly convert target adenines into inosines that are interpreted by cellular polymerases as guanines, offering therapeutic avenues for correcting pathogenic point mutations with minimal insertions or deletions. In contrast to conventional CRISPR nucleases that frequently stimulate error-prone non-homologous end joining pathways, base editors minimize unintended chromosomal translocations and extensive genomic rearrangements. Continuous protein engineering has expanded targeting scope through relaxed protospacer adjacent motif variants and optimized deaminase kinetics for high-fidelity cellular therapies.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "Permafrost thaw across the circum-Arctic boreal biome represents a potent positive feedback mechanism within global climate dynamics. Perennially frozen soils store an estimated one thousand four hundred billion metric tons of organic carbon accumulated from Pleistocene plant and animal biomass. As Arctic amplification accelerates atmospheric warming, the active thaw layer deepens, exposing previously inert organic matter to microbial decomposition. Under aerobic conditions, heterotrophic soil microbes oxidize thawed carbon into carbon dioxide. In waterlogged thermokarst wetlands and collapsed bogs, anoxic conditions favor methanogenic archaea that release methane, a greenhouse gas with substantially higher warming potential over decadal timescales. Thermokarst subsidence also triggers widespread ground destabilization, causing leaning drunken forests and severe structural damage to northern civil infrastructure. Furthermore, dissolved organic carbon leaching into Arctic river drainages alters estuarine chemistry, stimulating aquatic microbial respiration and accelerating carbon release into high-latitude maritime atmospheres.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "The engineering triumph of Roman monumental architecture was enabled by the development of opus caementicium, an extraordinary hydraulic concrete composed of volcanic pozzolana ash, slaked lime, and volcanic tuff aggregate. When mixed with water, reactive aluminosilicates in the volcanic ash from Pozzuoli reacted with calcium hydroxide, precipitating crystalline calcium-silicate-hydrate binder phases. Unlike modern Portland cement, ancient Roman concrete continued to gain compressive strength over centuries through post-curing mineral reactions. Seawater infiltration into maritime concrete structures dissolved volcanic aggregate minerals, precipitating rare interlocking crystals of aluminous tobermorite and phillipsite that prevented fracture propagation and enabled grand dome constructions like the Pantheon. The strategic inclusion of lighter aggregates like pumice near dome summits dramatically reduced structural deadweight while preserving compressive rigidity. This sophisticated understanding of material chemistry allowed Roman master builders to construct vast unreinforced coffered domes and monumental harbor moles that remain intact across two millennia of exposure.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "Deep brain stimulation is an established neuromodulatory surgical intervention utilized for alleviating motor symptoms in advanced Parkinson's disease and essential tremor. The clinical procedure involves neurosurgical stereotactic implantation of quadripolar depth electrodes into target subcortical nuclei, typically the subthalamic nucleus or the internal globus pallidus. An implanted pulse generator delivers continuous high-frequency electrical pulses that disrupt pathological synchronized beta-band oscillatory activity characteristic of dopamine-depleted basal ganglia circuitry. While the exact neurophysiological mechanisms remain subject to investigation, stimulation modulates local synaptic transmission, induces axonal block, and restores physiologically balanced motor information processing. Subsequent post-operative programming sessions customize electrical stimulation amplitudes, pulse widths, and contact configurations to maximize tremor suppression while minimizing speech impairments and cognitive side effects. Closed-loop responsive neurostimulation systems now leverage recorded local field potentials to dynamically adjust electrical delivery in direct response to real-time neural biomarkers.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "The integration of behavioral insights into public administration, often termed nudge theory, seeks to guide citizen choices in welfare-enhancing directions without mandating prohibitions or altering financial incentives. Popularized by Thaler and Sunstein, the approach alters choice architecture by leveraging default rules, social norms, and salience. In organ donation registries, transitioning from explicit opt-in to presumed consent defaults dramatically increases registration percentages because decision-makers exhibit inertia and perceive default options as implicit recommendations. Similarly, providing energy consumers with comparative normative feedback illustrating neighbors' electricity consumption induces lasting conservation behavior through descriptive social comparison. Critiques of nudge interventions question whether minor cognitive adjustments can resolve structural socioeconomic inequities, and highlight potential paternalistic overreach when choices are nudged covertly. Nonetheless, randomized controlled trials conducted by national behavioral insight teams confirm that low-cost informational framing reliably enhances tax compliance, pension savings, and preventive vaccination uptake across diverse populations.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "Gravitational lensing is an astrophysical consequence of general relativity wherein the gravitational field of a massive foreground object bends the spacetime trajectory of electromagnetic radiation emitted by a background source. According to Einstein's field equations, mass and energy curve surrounding geometry, causing photons to follow null geodesics around dense cosmic structures. Strong gravitational lensing occurs when a distant quasar aligns closely behind a massive elliptical galaxy or galaxy cluster, magnifying the background image and splitting it into multiple distinct arcs, rings, or Einstein crosses. Astronomers exploit these natural cosmic telescopes to investigate extremely distant infant galaxies and measure cosmic expansion rates through time-delay cosmography. Weak gravitational lensing produces slight statistical alignments in the shapes of background galaxies, enabling researchers to reconstruct large-scale three-dimensional dark matter distribution across cosmological filaments. Microlensing events caused by individual stars or exoplanets further permit the detection of non-luminous compact objects and rogue planetary bodies across the galactic disk.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "Metamorphic core complexes are remarkable geological structures that document intense crustal extension and ductile crustal unroofing. Formed during post-orogenic gravitational collapse, high-displacement low-angle normal detachment faults shear upper-crustal brittle fault blocks away from underlying mid-crustal ductile shear zones. The exhumed lower plate undergoes retrograde mylonitization and rapid tectonic decompression, preserving deep amphibolite-grade metamorphic fabrics beneath a sharp chlorite-breccia fault surface. Structural mapping and radiometric dating of lineated mylonites in the North American Basin and Range province confirm that tens of kilometers of crustal extension occurred within a few million years, fundamentally reshaping continental topography. Thermochronological analysis reveals rapid lower-plate cooling rates as detachment slip drew deep rocks toward surface temperatures. Associated syn-extensional plutonism and volcanism provided thermal weakening that facilitated continued crustal shearing and basin formation throughout modern metamorphic core terranes.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "The epistemology of disagreement investigates how rational agents should adjust their credences when encountering an epistemic peer who holds an incompatible belief despite evaluating the exact same evidentiary corpus. Equal-weight views argue that peer disagreement provides potent higher-order evidence of cognitive fallibility, obligating both parties to conciliate their beliefs by splitting the difference or suspending judgment. In contrast, steadfast perspectives assert that conciliating in every peer dispute leads to skepticism and self-refutation, insisting that agents may reasonably retain their original conclusions if their primary first-order evaluation of the arguments was rational. This theoretical debate carries vital implications for scientific controversy and political polarization. Intermediary views suggest that the degree of required belief adjustment depends on whether the dispute concerns basic perceptual judgments or complex inferential theories where comprehensive evidentiary appraisal is notoriously difficult. Exploring higher-order epistemic defeats illuminates how intellectual humility must be balanced against principled epistemic self-trust in collaborative academic research.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations.",
        "Superconducting transmon qubits are engineered circuit quantum electrodynamics devices that constitute leading hardware platforms for scalable quantum information processing. Built from lithographically patterned thin-film aluminum or niobium, transmons incorporate a Josephson tunnel junction shunted by a large auxiliary capacitor. The non-linear dissipationless inductance of the Josephson junction introduces anharmonicity into the circuit's energy level spectrum, allowing researchers to isolate the two lowest quantum energy levels as an effective computational qubit. Operating inside dilution refrigerators at temperatures below twenty millikelvin suppresses thermal excitations, permitting microwave pulse manipulation and dispersive dispersional readout via coupled superconducting coplanar waveguide resonators. The deliberate engineering of a high ratio between Josephson energy and charging energy makes transmons remarkably insensitive to charge noise fluctuations that hindered earlier charge qubits. Contemporary surface-code architectures couple multiple planar transmons through tunable bus couplers, facilitating fault-tolerant quantum error correction experiments on multi-qubit processor chips.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "The endosymbiotic hypothesis posits that eukaryotic cellular complexity originated from an ancient syntrophic partnership between an archaeal host and an alphaproteobacterial endosymbiont. Genomic sequencing of Asgard archaea has identified cellular machinery once considered unique to eukaryotes, such as actin-like cytoskeletal proteins and membrane-trafficking homologues, confirming that the host cell belonged to an archaeal lineage. Over evolutionary time, the engulfed alphaproteobacterial partner underwent radical reductive genome evolution, transferring the vast majority of its functional genes to the host nucleus while transforming into specialized double-membraned organelles dedicated to oxidative phosphorylation and ATP synthesis. The acquisition of this bioenergetic powerhouse relieved bioenergetic surface-area-to-volume constraints on cell size, facilitating dramatic expansion of eukaryotic genome size and gene expression complexity. Residual mitochondrial genomes and inner membrane cristae structures provide indelible molecular signatures of this ancient symbiotic merger that occurred roughly two billion years ago.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines.",
        "Tonogenesis describes the historical linguistic process through which non-tonal languages develop phonemic pitch contrasts across syllables. In East and Southeast Asian languages such as Vietnamese and early Middle Chinese, the loss of consonantal voicing contrasts in syllable onsets and laryngeal consonants in syllable codas induced compensatory pitch modifications on neighboring vowels. Voiced initial consonants lowered the intrinsic fundamental frequency of following vocalic sounds, whereas voiceless initials produced elevated pitch contours. Following the subsequent phonetic merger of initial consonants, these transient acoustic pitch differences became phonologized as distinctive tonal registers essential for lexical discrimination. Similarly, the loss of post-vocalic glottal stops typically generated rising tonal contours, whereas debuccalized post-vocalic fricatives produced falling intonation patterns. This phonological reanalysis illustrates how articulatory aerodynamic constraints can become systematized into intricate lexical pitch systems over centuries of generational language transmission.  Detailed empirical observations and methodological field studies continue to corroborate these theoretical interpretations across multiple independent research investigations. Moreover, systematic documentation and reproducible laboratory analyses remain foundational to scientific consensus across all academic disciplines."
    ]

    for i, o_text in enumerate(ORIGINALS, 1):
        fname = f"orig_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(o_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "c_original",
            "expected_plagiarism": "no",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"Fully original topic {i:02d}",
        })

    # --------------------------------------------------------------------------
    # GROUP D: 15 AI-Style Passages (Structured, formulaic, low burstiness)
    # --------------------------------------------------------------------------
    AI_PASSAGES = [
        "In recent years, the transition toward renewable energy sources has emerged as a cornerstone of sustainable global development. Furthermore, it is crucial to recognize that solar photovoltaic systems and wind turbines provide indispensable benefits for reducing carbon emissions. Moreover, the integration of smart electrical grids facilitates optimal power distribution across both urban and rural environments. Additionally, advancements in lithium-ion storage solutions ensure that intermittent generation does not jeopardize grid stability. Consequently, policymakers must cultivate collaborative frameworks to accelerate the adoption of clean energy alternatives. In conclusion, the multifaceted nature of energy transformation serves as a testament to humanity's ongoing dedication to climate resilience and technological progress. Moreover, continuous stakeholder engagement and transparent public-private partnerships will remain essential to streamline cross-border regulatory standards. As renewable technologies mature, proactive educational initiatives can empower local workforces to participate actively in sustainable green economies, ensuring lasting environmental stewardship and inclusive economic prosperity worldwide.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Artificial intelligence is poised to fundamentally revolutionize the landscape of contemporary healthcare delivery and clinical diagnostics. To begin with, machine learning algorithms can analyze vast volumes of complex medical imagery with remarkable accuracy and efficiency. Furthermore, predictive models assist clinicians by identifying early indicators of patient deterioration in intensive care units. Moreover, natural language processing tools effectively streamline clinical documentation, thereby reducing administrative burdens on medical professionals. However, it is essential to emphasize that ethical considerations, such as patient privacy and algorithmic transparency, remain paramount. In conclusion, the harmonious synergy between human expertise and automated intelligence represents a promising frontier for modern clinical medicine. Additionally, standardized protocols for algorithmic validation and clinical auditing are necessary to foster physician trust and patient safety. Ultimately, ongoing interdisciplinary collaboration between medical practitioners, computer scientists, and health policy ethicists will ensure that artificial intelligence enhances compassionate and equitable healthcare delivery on a global scale.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "The integration of digital pedagogical platforms has reshaped instructional delivery and student engagement across modern academic institutions. First and foremost, digital learning environments afford unprecedented flexibility, enabling learners to access instructional materials irrespective of geographical barriers. Furthermore, interactive multimedia resources foster collaborative inquiry and personalized educational trajectories for diverse student cohorts. Additionally, automated evaluation mechanisms provide rapid diagnostic feedback, allowing instructors to intervene proactively when comprehension challenges arise. Nonetheless, educators must remain vigilant regarding equitable technological access and digital literacy divides. In conclusion, embracing modern technological methodologies while preserving robust pedagogical foundations is essential for preparing students for future professional landscapes. Furthermore, institutional investments in modern network infrastructure and teacher training programs are crucial to ensure that technological advancements enhance educational equity rather than exacerbate socioeconomic disparities. By balancing technological efficiency with empathetic mentorship, educational institutions can foster inclusive learning environments that inspire lifelong curiosity and academic excellence.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Sustainable urban transportation systems are critical components of resilient metropolitan infrastructure in the twenty-first century. To begin with, high-capacity public transit networks, such as electric light rail and bus rapid transit, alleviate severe traffic congestion. Furthermore, dedicated cycling infrastructure and pedestrian corridors encourage active commuting practices that improve public health outcomes. Moreover, the electrification of municipal transit fleets substantially curtails localized atmospheric particulate pollution and greenhouse gas emissions. Therefore, urban planners must prioritize integrated transit-oriented development frameworks that connect residential districts with employment hubs. In conclusion, comprehensive mobility initiatives play a pivotal role in fostering vibrant, economically robust, and environmentally sustainable urban communities. In addition, the integration of intelligent traffic management algorithms and real-time public transit tracking apps maximizes passenger convenience while minimizing operational overhead. Moving forward, collaborative urban governance and sustained public financing will be essential to realize truly resilient, emission-free, and equitable transportation networks across modern global cities.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Safeguarding modern financial infrastructure against sophisticated cyber threats represents a critical priority for contemporary institutional governance. First of all, distributed denial-of-service incursions and credential theft schemes pose severe risks to commercial banking availability and consumer confidence. Furthermore, the adoption of zero-trust architectural principles ensures that all network entities undergo rigorous authentication and continuous authorization. Moreover, end-to-end cryptographic encryption standards protect confidential monetary transaction records across public communication channels. Additionally, proactive penetration testing and vulnerability assessments enable security teams to remediate systemic weaknesses prior to exploitation. In conclusion, cultivating resilient institutional cyber defenses is indispensable for maintaining monetary stability and safeguarding digital global economies. Moreover, establishing collaborative threat intelligence sharing alliances among international regulatory agencies and financial institutions enables rapid coordinated responses to emerging cyber vulnerabilities. By fostering a pervasive culture of cyber awareness and rigorous compliance, organizations can maintain enduring customer trust in an increasingly interconnected global digital economy.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Sustainable freshwater management constitutes an urgent environmental imperative in the face of escalating global climatic shifts. To commence with, agricultural irrigation systems consume substantial quantities of freshwater resources worldwide through conventional flood methodologies. Furthermore, precision drip irrigation technologies drastically curtail evaporative water losses while optimizing agricultural crop productivity. Moreover, municipal water recycling plants and advanced membrane desalination facilities augment regional water security for growing urban populations. Additionally, watershed restoration projects and wetland conservation efforts preserve natural filtration ecosystems that recharge underlying aquifers. In conclusion, balanced water governance combining technological innovation with rigorous conservation policies is essential for guaranteeing equitable resource access for future generations. Furthermore, adopting integrated circular water management models that combine stormwater harvesting with decentralized graywater treatment systems reduces reliance on municipal reservoir reserves. Consequently, comprehensive cross-sector water governance is imperative to preserve aquatic habitats, mitigate drought vulnerability, and safeguard human flourishing in an unpredictable global climate.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Modern supply chain networks require sophisticated logistical coordination to satisfy the dynamic requirements of global e-commerce markets. First and foremost, real-time tracking technologies and automated warehousing facilities enhance fulfillment velocity and inventory precision. Furthermore, predictive analytics models enable supply chain managers to anticipate seasonal demand fluctuations and mitigate shipping bottlenecks. Moreover, diversifying supplier relationships across multiple geographic regions reduces vulnerability to unexpected geopolitical disruptions and severe natural disasters. Consequently, enterprises that embrace agile and transparent logistics operations secure substantial competitive advantages in contemporary commercial landscapes. In conclusion, the strategic optimization of supply chains remains a vital foundation for sustaining international trade and consumer satisfaction. Additionally, embracing automated end-to-end provenance verification and decentralized ledger technologies enhances compliance auditing and product traceability across complex multi-tier supply chains. Moving forward, prioritizing organizational agility, ethical sourcing, and operational transparency will remain central to driving long-term profitability and sustainable commercial success.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "The widespread proliferation of microplastic contaminants poses severe ecological threats to marine biodiversity and aquatic food webs worldwide. To start with, synthetic polymer debris fragments into microscopic particles through ultraviolet photodegradation and mechanical wave abrasion. Furthermore, marine organisms across multiple trophic levels ingest microplastics, leading to internal physical blockages and diminished reproductive fitness. Moreover, microplastic surfaces adsorb persistent organic pollutants from surrounding seawater, magnifying toxic chemical bioaccumulation within marine life. Therefore, international regulatory frameworks must establish strict prohibitions on non-essential plastic packaging and enhance wastewater filtration protocols. In conclusion, addressing marine plastic pollution requires coordinated international intervention to safeguard oceanic habitats and long-term ecological balance. Moreover, consumer education campaigns aimed at encouraging reusable packaging materials and promoting responsible post-consumer waste management can substantially diminish plastic leakage into coastal waterways. Ultimately, a holistic synthesis of technological innovation, legislative mandates, and international treaty enforcement is crucial to safeguard marine life for generations to come.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Cloud computing architectures provide scalable and reliable computational infrastructure for contemporary enterprise software applications and data analytics. First of all, on-demand resource provisioning allows organizations to expand or contract server capacity rapidly without incurring excessive capital expenditures. Furthermore, microservices architectures and containerization platforms facilitate modular software development, deployment velocity, and fault tolerance. Moreover, distributed cloud storage frameworks incorporate redundant data replication protocols that guarantee uninterrupted operational continuity during hardware failures. Consequently, modern businesses can accelerate technological innovation while optimizing recurring infrastructure maintenance expenditures. In conclusion, the pervasive transition toward cloud-native ecosystems underscores the transformative role of distributed computing in contemporary digital operations. Furthermore, organizations must prioritize comprehensive multi-cloud strategies and zero-trust security postures to mitigate vendor lock-in risks and protect mission-critical enterprise workloads. By continuously optimizing cloud infrastructure through automated telemetry and artificial intelligence, modern enterprises can achieve unparalleled operational agility and sustainable commercial growth.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Agricultural biotechnology offers promising technological avenues for augmenting global crop yields and addressing widespread food security challenges. To begin with, genetic engineering techniques facilitate the development of staple crops exhibiting enhanced resistance to severe drought and pest infestations. Furthermore, biofortification initiatives augment the nutritional composition of essential grains by introducing vital micronutrients like vitamin A and iron. Moreover, reducing dependence on synthetic chemical pesticides mitigates environmental contamination and protects beneficial insect populations within agricultural ecosystems. Nonetheless, transparent regulatory assessments and comprehensive environmental impact evaluations remain imperative to address valid societal concerns. In conclusion, responsible deployment of agricultural biotechnology represents an important instrument for establishing resilient and sustainable global food systems. Moreover, collaborative public-private partnerships and targeted international development grants can facilitate equitable technology transfer to resource-constrained agricultural communities worldwide. Ultimately, integrating biotechnological innovations within broader regenerative agroecological practices will be crucial to establishing resilient global food security amidst evolving environmental pressures.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "The widespread adoption of remote work paradigms has profoundly altered organizational culture, employee productivity, and traditional workplace dynamics. First and foremost, virtual collaboration platforms and asynchronous communication tools enable geographically dispersed teams to execute complex projects seamlessly. Furthermore, employees benefit from enhanced schedule flexibility and reduced commuting stress, which often translates into heightened job satisfaction. Moreover, organizations can access international talent pools without the logistical constraints associated with physical corporate office spaces. However, corporate leaders must intentionally cultivate cohesive team cultures to prevent professional isolation and maintain collective organizational alignment. In conclusion, navigating the nuances of distributed work environments is vital for building resilient and forward-thinking corporate enterprises. Additionally, executive leadership must establish clear boundaries regarding working hours and encourage asynchronous collaboration to prevent employee burnout and maintain long-term professional well-being. Ultimately, organizations that cultivate intentional, transparent, and empathetic workplace cultures will thrive in attracting and retaining top international talent.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Green hydrogen energy is increasingly recognized as a pivotal chemical carrier for decarbonizing heavy industrial sectors and long-haul transportation. To commence with, water electrolysis powered by dedicated renewable energy installations generates zero-emission hydrogen gas suitable for thermal industrial processes. Furthermore, hydrogen fuel cell vehicles offer rapid refueling capabilities and extended operational ranges compared to heavy battery-electric alternatives. Moreover, utilizing surplus solar and wind electricity to produce hydrogen enables seasonal chemical energy storage across extended duration cycles. Consequently, strategic public infrastructure investments in hydrogen pipelines and storage caverns are essential to unlock commercial viability. In conclusion, renewable hydrogen represents a multifaceted technological solution for achieving ambitious international net-zero emissions targets. Furthermore, international harmonization of safety protocols and carbon accounting methodologies will be vital to accelerate cross-border hydrogen trade and stimulate global private investment. By fostering cross-industry alliances and supportive regulatory frameworks, global economies can successfully harness green hydrogen to realize comprehensive and durable industrial decarbonization.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Recent advancements in commercial aerospace engineering have initiated a transformative era of low-Earth orbit satellite constellation deployment and space exploration. First of all, reusable launch vehicles have dramatically reduced orbital payload delivery costs, facilitating unprecedented access to outer space. Furthermore, mega-constellations of compact communication satellites deliver high-speed broadband internet connectivity to underserved and remote geographic regions. Moreover, advanced orbital observation platforms provide vital real-time environmental data for monitoring global deforestation, wildfire progression, and agricultural productivity. Nevertheless, the proliferation of space debris necessitates the implementation of binding international orbital traffic management guidelines. In conclusion, modern aerospace innovation promises profound socioeconomic benefits while underscoring the necessity of sustainable orbital space stewardship. Moreover, international treaties must be modernized to prevent orbital conflict and ensure equitable spectrum allocation for emerging aerospace nations. By balancing rapid commercial innovation with proactive environmental stewardship, humanity can unlock unprecedented scientific discoveries while preserving the peaceful sustainability of outer space.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "Quantum computing architectures represent an extraordinary paradigm shift in information processing capable of solving specific problems intractable for classical supercomputers. To begin with, quantum phenomena like superposition and entanglement enable quantum processors to explore vast mathematical solution spaces simultaneously. Furthermore, quantum simulation algorithms promise breakthrough discoveries in molecular modeling, battery chemistry, and high-temperature superconductor synthesis. Moreover, Shor's algorithm demonstrates the theoretical potential of fault-tolerant quantum hardware to factor large integers, prompting post-quantum cryptography transitions. However, overcoming quantum decoherence through cryogenic engineering and fault-tolerant quantum error correction remains a formidable technical challenge. In conclusion, the relentless pursuit of scalable quantum computing stands as a monument to modern scientific ingenuity and theoretical physics. Furthermore, establishing open-source software libraries and quantum hardware benchmarking standards will be essential to nurture global scientific collaboration and train future quantum engineers. Ultimately, the deliberate development of quantum computing infrastructure promises to revolutionize computational science, materials engineering, and cryptographic security in profound ways.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience.",
        "The development and industrial adoption of biodegradable polymers represent an essential transition toward establishing circular global material economies. First and foremost, bio-based plastics derived from renewable feedstocks such as cornstarch and sugarcane reduce dependence on finite petrochemical reserves. Furthermore, compostable polyesters like polylactic acid naturally degrade into harmless carbon dioxide, water, and organic biomass under controlled conditions. Moreover, integrating advanced enzymatic additives facilitates ambient marine biodegradation, mitigating ecological harms caused by discarded packaging litter. Consequently, governments should implement supportive public procurement policies and extended producer responsibility mandates to accelerate market adoption. In conclusion, advancing sustainable material sciences is critical for curbing plastic waste and preserving delicate terrestrial and marine ecosystems. Moreover, cross-sector industrial standards and municipal composting infrastructure investments are essential to ensure that biodegradable materials decompose efficiently without contaminating conventional mechanical recycling streams. By combining material science innovations with circular economic principles, societies can establish resilient industrial cycles that preserve natural resources.  In conclusion, prioritizing comprehensive strategic frameworks and fostering transparent cross-disciplinary dialogue will remain pivotal for achieving sustainable outcomes and long-term societal resilience."
    ]

    for i, a_text in enumerate(AI_PASSAGES, 1):
        fname = f"ai_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(a_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "d_ai_style",
            "expected_plagiarism": "no",
            "expected_ai": "yes",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"AI-style text {i:02d}",
        })

    # --------------------------------------------------------------------------
    # GROUP E: 10 Human Non-Native (ESL) Passages (False Positive Check)
    # --------------------------------------------------------------------------
    NON_NATIVE_PASSAGES = [
        "When I was small boy in my village, we did not have any computer or internet. My father was working in post office and he bring home one old typewriter machine. I was typing letters all day and very happy. Later when I go to university in big city, I saw real computer first time. It was very big and noisy computer with green screen. The teacher tell us do not touch button because it is expensive. But one day computer stop working because dust inside. I open the metal cover and clean everything with small brush. When teacher come back, computer working very good and teacher give me one big smile. That is why I want study computer science today. My father was so proud that day. He tell everyone in post office that his son is computer genius, even though I only clean dust with brush. Next month he buy me old mathematics book and English dictionary from second hand shop. I study hard every night by candlelight. That small old computer changed my whole life and gave me dream to become programmer.",
        "Every day morning I must wake up early at five o'clock to take the bus to university campus. In my city the bus is always full with many people and you cannot find any seat. Sometimes the driver drive very fast and people almost fall down when bus make sharp turn. One time raining very hard and roof of bus had water leaking on my jacket and my books. But I like taking the bus because I can see many interesting people. There is old man who always read history newspaper and grandmother who give candies to crying children. Even when bus is late for forty minutes, it is good time for me to memorize English vocabulary for test. When I reach university after long bus journey, I always feel tired but energetic inside my heart. My friends wait for me at entrance gate with warm tea and biscuits. We talk about professors and assignments before morning lecture begin. Even with all the noise and rain, that bus ride taught me how to be patient and appreciate simple everyday moments of city life.",
        "In my country food is very important part of life and family gathering. When my grandmother cook her famous lentil soup, the whole house smell so delicious with garlic and cumin spices. She always tell me that real cooking need patience and love, not just fast boiling. You must cut onion very small and fry slowly until golden color. Then add fresh tomatoes from garden and special dried herbs. When all my cousins and uncles sit around big table, we eat with fresh bread and talk about everything. Nobody look at smartphone when eating. Now I live in student dormitory abroad and I try cook same soup, but taste is never same like grandmother house. Last week I invite three friends from my laboratory to my small dormitory room. I cook lentil soup using same spices my mother sent by postal package. When we sit together on floor and eat hot soup, my friends smile and say it is best food they ate all month. It made me feel like I am back home with my family, even though my real home is thousands of miles away.",
        "When I first arrive to this country for my masters degree, the cold weather was big shock for me. In my hometown near equator, temperature is always thirty degrees and sunny all year. Here in winter time, sun go down at four in afternoon and sky is grey for many weeks. First time I see snow falling, I run outside without winter boots because I was so excited like small child. Snow was soft and pretty, but after five minutes my feet were freezing cold like ice blocks. My roommate from Poland laugh and lend me his big jacket and warm gloves. Now I learn to drink hot ginger tea and wear two pairs of woolen socks when going to library. By the time spring came, everything changed in wonderful way. Green grass start growing under melting snow, and cherry blossoms open on trees near campus lake. People were smiling on street and sitting outside in sunshine. I realized that surviving cold winter make spring season feel ten times more beautiful. Now I am not afraid of winter anymore, because I know spring always return.",
        "Learning English language is very difficult thing for me because grammar rules are so different from my mother language. In my language, we do not have articles like 'a' and 'the', so I always forget to write them in my essay papers. Also pronunciation of words like 'thought' and 'through' look same but sound completely different. My English teacher is very patient woman. She tell me that making mistakes is normal part of learning and I should not be shy to speak in classroom. I practice speaking with mirror every night and watch English movies with subtitles. Slowly day by day I feel my confidence getting better and I can talk with international students. Yesterday was very special day because I give fifteen minute scientific presentation in front of fifty professors and students. In beginning my hands were shaking and voice was quiet. But then I remember advice from my teacher and look at kind faces in audience. I speak clearly and answer three difficult questions. After presentation, my professor shake my hand and say \"well done\". I was so proud of myself.",
        "During summer holidays my grandfather always take me to his vegetable garden behind the village house. He was planting red tomatoes, green cucumbers, and big sweet watermelons. He teach me how to pull weeds without hurting the vegetable roots. He said soil is alive and if you take care of earth, earth will give you good food to eat. When sun was very hot at noon, we sit under walnut tree and drink cold water from mountain well. He tell me stories about how village was fifty years ago when there was no electricity. Those quiet afternoons with my grandfather are my most precious memories from childhood. Now that I am grown man living in foreign country, I still think about those summer days in garden with my grandfather. Whenever I feel stressed with research papers and computer screens, I go to city park and sit quietly under oak tree. I close my eyes and remember smell of fresh tomato leaves and sound of wind through walnut branches. It always brings peace back into my soul.",
        "My first job when I was eighteen was working as cashier in grocery supermarket. In the beginning I was very nervous because numbers on cash register must match exactly with money in drawer at end of night. If money is missing, manager make you pay from your salary. Some customers were in big hurry and shout when barcode scanner not work fast. But many old people were very friendly and chat with me about their grandchildren while I pack their bags. Working long standing hours make my legs hurt so much, but when I receive my first monthly paycheck, I was so proud. I buy winter coat for my mother with that money. Two years later, I was promoted to shift supervisor and train new student cashiers who were nervous just like me on my first day. I always treat them with kindness and patience. That simple grocery store job taught me more about human psychology and hard work than many university textbooks. It gave me respect for all working people who earn honest living with their sweat.",
        "Moving to student dormitory was first time I live away from my family. On first night I feel very lonely in small room with empty white walls and narrow bed. I can hear cars outside on street and strange noises in hallway. But next morning I meet my neighbor who come from Brazil. He invite me to drink coffee and show me where is laundry machine in basement. We share our pots and frying pans in common kitchen and teach each other words in our languages. Now our dormitory floor feel like one big international family where everyone help each other when exams are difficult. Two years have passed since that first lonely night, and tomorrow we all graduate together from university. We pack our bags and give each other long hugs with tears in our eyes. This dormitory was not just building made from concrete; it was our home where we grow up and learn to care for people from different cultures. Those friendships will stay in my heart forever.",
        "My bicycle was second hand and very cheap when I bought it from older student who graduated. Chain was very rusty and gears make terrible squeaking noise when riding uphill. One Saturday afternoon I decide to fix it myself with simple tools. I clean the metal chain with kerosene and put machine oil on all moving wheels. I change the rubber brake pads because old ones were completely worn down. My hands were black with grease, but when I test ride on street, bicycle was moving smooth and silent like new one. It gave me great feeling of satisfaction to repair something with my own hands instead of buying new. Since that day, I never pay money to bicycle repair shop again. I fix bicycles for all my dormitory friends when their tires get punctured or chains get loose. They bring me hot coffee and homemade cakes to say thank you. Repairing old mechanical things taught me that patience and care can make old broken objects useful and beautiful again for many years.",
        "Final examination week at university is always very stressful time for all engineering students. We stay in university library until midnight drinking strong black coffee and solving mathematics problem sets. Sometimes equations are so long and complicated that my head feel like burning. But my study group friends are very supportive. When one student not understand thermodynamics formula, another student explain on whiteboard until everyone understand. After last exam finish on Friday afternoon, we all go together to cheap pizza restaurant and laugh loudly about how worried we were. That feeling of relief after hard work is best feeling in world. Looking back at university years, those late night study sessions in library are memories I cherish most deeply. We were young, tired, and scared of failing, but we faced every difficult academic challenge together as true friends. That shared struggle forged bonds between us that time and distance will never break, and today we all work successfully as professional engineers.  When I think about these important memories now, I feel very grateful for all people who help me and teach me good lessons in my journey."
    ]

    for i, nn_text in enumerate(NON_NATIVE_PASSAGES, 1):
        fname = f"non_native_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(nn_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "e_non_native",
            "expected_plagiarism": "no",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"Non-native ESL human writer {i:02d}",
        })

    # --------------------------------------------------------------------------
    # GROUP F: 10 Spinner Documents + 10 Clean Controls
    # --------------------------------------------------------------------------
    SPINNERS = [
        # Spin 01: Artificial intelligence -> counterfeit consciousness, profound learning
        (
            "The recent progression of counterfeit consciousness has revolutionized computer vision and computational linguistics. "
            "Modern researchers apply profound learning algorithms to process extensive imagery datasets with unprecedented velocity. "
            "Instead of standard parameter tuning, engineers implement arbitrary timberland classifiers to establish baseline predictive performance. "
            "Furthermore, colossal telemetry gathered from interconnected edge devices informs real-time inference optimization across distributed clusters. "
            "In conclusion, adopting counterfeit consciousness methodologies within vital organization systems enables enhanced operational resilience and efficiency."
        ),
        # Spin 02: Big data -> colossal telemetry, machine learning -> engine learning
        (
            "Handling colossal telemetry generated by international sensor grids presents substantial computational infrastructure challenges for enterprise architects. "
            "Modern analytical software employs engine learning pipelines to identify latent statistical anomalies within high-throughput message queues. "
            "When relational databases fail under heavy loads, developers transition toward arbitrary timberland storage schemas and distributed document stores. "
            "Consequently, counterfeit consciousness algorithms evaluate streaming records concurrently, extracting actionable strategic intelligence from raw data streams. "
            "These technological capabilities empower organizations to optimize resource allocation across complex industrial manufacturing environments."
        ),
        # Spin 03: Neural network -> neural organization, deep learning -> profound learning
        (
            "Training complex neural organization architectures requires substantial graphical acceleration units and specialized numeric optimization routines. "
            "Over the past decade, profound learning paradigms have displaced classical rule-based heuristics across voice recognition and speech synthesis domains. "
            "Researchers frequently compare gradient descent convergence against arbitrary timberland baseline benchmarks to measure relative performance margins. "
            "Moreover, counterfeit consciousness models deployed on resource-constrained embedded microcontrollers necessitate aggressive quantization and structural weight pruning. "
            "Such engineering refinements guarantee reliable real-time response latency across critical automotive and industrial control systems."
        ),
        # Spin 04: Random forest -> arbitrary timberland
        (
            "Predictive modeling on tabular clinical datasets frequently benefits from deploying arbitrary timberland ensemble algorithms over single decision trees. "
            "By aggregating individual tree predictions through majority voting, arbitrary timberland methods mitigate variance without inflating training bias. "
            "Furthermore, counterfeit consciousness platforms leverage these ensemble frameworks to forecast patient readmission probabilities within emergency departments. "
            "Colossal telemetry streaming from physiological monitoring equipment supplies continuous observational inputs to the underlying inference engine. "
            "Ultimately, combining clinical domain expertise with profound learning algorithms enhances diagnostic accuracy and therapeutic intervention planning."
        ),
        # Spin 05: Vital organization -> important organization / vital organization
        (
            "Maintaining secure cryptographic handshakes represents a foundational imperative for any vital organization operating digital transactional platforms. "
            "When malicious threat actors attempt network intrusion, counterfeit consciousness defensive agents analyze packet headers to detect suspicious patterns. "
            "Colossal telemetry transmitted across enterprise boundaries undergoes automated behavioral anomaly screening powered by profound learning detectors. "
            "Additionally, arbitrary timberland regression estimators forecast bandwidth saturation risks during distributed denial-of-service attack scenarios. "
            "Implementing these defense-in-depth measures ensures continuous operational continuity for critical institutional infrastructure worldwide."
        ),
        # Spin 06:
        (
            "Natural language understanding has experienced remarkable breakthroughs through the introduction of counterfeit consciousness transformer networks. "
            "Profound learning representations capture intricate syntactic dependencies and semantic nuances across multilingual text corpora effectively. "
            "Meanwhile, traditional data mining approaches relying on arbitrary timberland classifiers remain popular for lightweight tabular prediction tasks. "
            "Colossal telemetry produced by web crawling services provides the foundational pre-training corpus required for large language model convergence. "
            "Ensuring algorithmic fairness and robust verification standards across every vital organization deploying these models remains an ongoing scientific priority."
        ),
        # Spin 07:
        (
            "Autonomous robotic navigation across unstructured agricultural terrain requires real-time sensor fusion and high-speed motor control algorithms. "
            "Autonomous tractors utilize counterfeit consciousness perception stacks to distinguish harvestable crops from invasive weed vegetation reliably. "
            "Profound learning convolutional backbones process high-resolution multispectral imagery, feeding obstacle coordinates into trajectory planning modules. "
            "Simultaneously, colossal telemetry from localized wheel encoders and satellite receivers updates terrain elevation maps on edge processors. "
            "Deploying arbitrary timberland decision models ensures that the vehicle can respond safely to unexpected physical hazards without human intervention."
        ),
        # Spin 08:
        (
            "Computational biology relies increasingly upon profound learning to decipher complex protein folding structures from amino acid sequence data. "
            "Where classical molecular dynamics simulations required months of supercomputer runtime, counterfeit consciousness models predict atomic coordinates in minutes. "
            "Colossal telemetry generated by cryo-electron microscopy facilities supplies high-resolution structural benchmarks that validate predicted conformations. "
            "Furthermore, arbitrary timberland scoring functions assess binding affinities between candidate small molecules and targeted disease receptors. "
            "These computational breakthroughs significantly accelerate early-stage drug discovery pipelines across modern biopharmaceutical research institutions."
        ),
        # Spin 09:
        (
            "The expansion of smart electrical power grids demands sophisticated forecasting algorithms to balance intermittent renewable energy generation. "
            "Counterfeit consciousness forecasting engines analyze historical meteorological data alongside real-time grid load measurements. "
            "Profound learning recurrent networks forecast solar irradiance and wind velocity twenty-four hours in advance with minimal variance. "
            "Simultaneously, colossal telemetry gathered from smart utility meters enables localized demand-response pricing mechanisms. "
            "Operating an arbitrary timberland ensemble allows utility dispatchers to schedule reserve generator capacity proactively, preventing brownouts."
        ),
        # Spin 10:
        (
            "Financial institutions deploy counterfeit consciousness algorithms to detect money laundering transactions and credit card fraud in real time. "
            "Profound learning graph architectures analyze transaction networks to identify hidden shell company relationships and suspicious transfer loops. "
            "Colossal telemetry originating from point-of-sale terminals around the world flows into centralized risk assessment clusters for instant evaluation. "
            "Complementary arbitrary timberland classification models evaluate transaction amounts, merchant categories, and geolocation discrepancies. "
            "Protecting customer assets while minimizing false-positive transaction declines represents a paramount operational objective for every vital organization."
        ),
    ]

    SPINNER_CONTROLS = [
        # Ctrl 01: Standard AI and deep learning text
        (
            "The recent progression of artificial intelligence has revolutionized computer vision and computational linguistics. "
            "Modern researchers apply deep learning algorithms to process extensive imagery datasets with unprecedented velocity. "
            "Instead of standard parameter tuning, engineers implement random forest classifiers to establish baseline predictive performance. "
            "Furthermore, massive telemetry gathered from interconnected edge devices informs real-time inference optimization across distributed clusters. "
            "In conclusion, adopting artificial intelligence methodologies within critical organization systems enables enhanced operational resilience and efficiency."
        ),
        # Ctrl 02: Big data and machine learning text
        (
            "Handling massive telemetry generated by international sensor grids presents substantial computational infrastructure challenges for enterprise architects. "
            "Modern analytical software employs machine learning pipelines to identify latent statistical anomalies within high-throughput message queues. "
            "When relational databases fail under heavy loads, developers transition toward distributed storage schemas and non-relational document stores. "
            "Consequently, artificial intelligence algorithms evaluate streaming records concurrently, extracting actionable strategic intelligence from raw data streams. "
            "These technological capabilities empower organizations to optimize resource allocation across complex industrial manufacturing environments."
        ),
        # Ctrl 03: Neural networks and deep learning
        (
            "Training complex neural network architectures requires substantial graphical acceleration units and specialized numeric optimization routines. "
            "Over the past decade, deep learning paradigms have displaced classical rule-based heuristics across voice recognition and speech synthesis domains. "
            "Researchers frequently compare gradient descent convergence against random forest baseline benchmarks to measure relative performance margins. "
            "Moreover, artificial intelligence models deployed on resource-constrained embedded microcontrollers necessitate aggressive quantization and structural weight pruning. "
            "Such engineering refinements guarantee reliable real-time response latency across critical automotive and industrial control systems."
        ),
        # Ctrl 04: Random forest algorithms
        (
            "Predictive modeling on tabular clinical datasets frequently benefits from deploying random forest ensemble algorithms over single decision trees. "
            "By aggregating individual tree predictions through majority voting, random forest methods mitigate variance without inflating training bias. "
            "Furthermore, artificial intelligence platforms leverage these ensemble frameworks to forecast patient readmission probabilities within emergency departments. "
            "Extensive telemetry streaming from physiological monitoring equipment supplies continuous observational inputs to the underlying inference engine. "
            "Ultimately, combining clinical domain expertise with deep learning algorithms enhances diagnostic accuracy and therapeutic intervention planning."
        ),
        # Ctrl 05: Cybersecurity in organizations
        (
            "Maintaining secure cryptographic handshakes represents a foundational imperative for any essential organization operating digital transactional platforms. "
            "When malicious threat actors attempt network intrusion, artificial intelligence defensive agents analyze packet headers to detect suspicious patterns. "
            "High-volume telemetry transmitted across enterprise boundaries undergoes automated behavioral anomaly screening powered by deep learning detectors. "
            "Additionally, random forest regression estimators forecast bandwidth saturation risks during distributed denial-of-service attack scenarios. "
            "Implementing these defense-in-depth measures ensures continuous operational continuity for critical institutional infrastructure worldwide."
        ),
        # Ctrl 06: Natural language understanding
        (
            "Natural language understanding has experienced remarkable breakthroughs through the introduction of modern transformer neural networks. "
            "Deep learning representations capture intricate syntactic dependencies and semantic nuances across multilingual text corpora effectively. "
            "Meanwhile, traditional data mining approaches relying on random forest classifiers remain popular for lightweight tabular prediction tasks. "
            "Extensive text corpora produced by web crawling services provide the foundational pre-training data required for large language model convergence. "
            "Ensuring algorithmic fairness and robust verification standards across every public organization deploying these models remains an ongoing scientific priority."
        ),
        # Ctrl 07: Autonomous robotics
        (
            "Autonomous robotic navigation across unstructured agricultural terrain requires real-time sensor fusion and high-speed motor control algorithms. "
            "Autonomous tractors utilize computer vision perception stacks to distinguish harvestable crops from invasive weed vegetation reliably. "
            "Deep learning convolutional backbones process high-resolution multispectral imagery, feeding obstacle coordinates into trajectory planning modules. "
            "Simultaneously, location telemetry from localized wheel encoders and satellite receivers updates terrain elevation maps on edge processors. "
            "Deploying random forest decision models ensures that the vehicle can respond safely to unexpected physical hazards without human intervention."
        ),
        # Ctrl 08: Computational biology
        (
            "Computational biology relies increasingly upon deep learning to decipher complex protein folding structures from amino acid sequence data. "
            "Where classical molecular dynamics simulations required months of supercomputer runtime, modern machine learning models predict atomic coordinates in minutes. "
            "Structural telemetry generated by cryo-electron microscopy facilities supplies high-resolution benchmarks that validate predicted conformations. "
            "Furthermore, random forest scoring functions assess binding affinities between candidate small molecules and targeted disease receptors. "
            "These computational breakthroughs significantly accelerate early-stage drug discovery pipelines across modern biopharmaceutical research institutions."
        ),
        # Ctrl 09: Smart grids
        (
            "The expansion of smart electrical power grids demands sophisticated forecasting algorithms to balance intermittent renewable energy generation. "
            "Advanced forecasting engines analyze historical meteorological data alongside real-time grid load measurements. "
            "Deep learning recurrent networks forecast solar irradiance and wind velocity twenty-four hours in advance with minimal variance. "
            "Simultaneously, detailed telemetry gathered from smart utility meters enables localized demand-response pricing mechanisms. "
            "Operating a random forest ensemble allows utility dispatchers to schedule reserve generator capacity proactively, preventing brownouts."
        ),
        # Ctrl 10: Fraud detection
        (
            "Financial institutions deploy machine learning algorithms to detect money laundering transactions and credit card fraud in real time. "
            "Deep learning graph architectures analyze transaction networks to identify hidden shell company relationships and suspicious transfer loops. "
            "Transaction telemetry originating from point-of-sale terminals around the world flows into centralized risk assessment clusters for instant evaluation. "
            "Complementary random forest classification models evaluate transaction amounts, merchant categories, and geolocation discrepancies. "
            "Protecting customer assets while minimizing false-positive transaction declines represents a paramount operational objective for every major bank."
        ),
    ]

    for i, s_text in enumerate(SPINNERS, 1):
        fname = f"spinner_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(s_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "f_spinner",
            "expected_plagiarism": "no",
            "expected_ai": "no",
            "expected_spinner": "yes",
            "expected_collusion_group": "none",
            "notes": f"Tortured phrases document {i:02d}",
        })

    for i, c_text in enumerate(SPINNER_CONTROLS, 1):
        fname = f"spinner_ctrl_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(c_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "f_spinner_control",
            "expected_plagiarism": "no",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"Clean control without tortured phrases {i:02d}",
        })

    # --------------------------------------------------------------------------
    # GROUP G: 8 Synthetic Class Submissions (Two colluding pairs + 4 independent)
    # --------------------------------------------------------------------------
    SHARED_PASSAGE_A = (
        "B-tree index structures maintain balanced hierarchical search trees where all leaf nodes reside at equal depths. "
        "Each internal node accommodates multiple keys and child pointers, achieving high branching factors that minimize disk input-output operations. "
        "When keys are inserted into saturated nodes, split operations propagate upward, preserving strict logarithmic search guarantees."
    )
    SHARED_PASSAGE_B = (
        "Hash indexes map arbitrary search keys directly into bucket locations using deterministic hash functions. "
        "While hash indexes provide constant-time point lookup efficiency, they cannot support range scan queries efficiently because hash order does not preserve key ordering. "
        "Consequently, relational database systems prioritize B-tree architectures for general-purpose transactional workloads."
    )

    CLASS_SUBS = [
        # Sub 01 (Colluding Pair 1)
        (
            "Database Indexing Analysis: Structural Efficiency in Modern DBMS.\n\n"
            f"{SHARED_PASSAGE_A}\n\n"
            f"{SHARED_PASSAGE_B}\n\n"
            "In conclusion, modern database systems evaluate storage trade-offs between indexing read performance and write penalty costs during heavy update operations."
        ),
        # Sub 02 (Colluding Pair 1)
        (
            "Evaluation of Index Architectures in Relational Database Engines.\n\n"
            f"{SHARED_PASSAGE_A}\n\n"
            f"{SHARED_PASSAGE_B}\n\n"
            "To summarize our findings, B-tree indexes continue to dominate database storage layers due to their robust predictable performance for mixed query workloads."
        ),
        # Sub 03 (Colluding Pair 2)
        (
            "LSM Trees versus Traditional In-Place Indexing Mechanisms.\n\n"
            "Log-Structured Merge trees buffer incoming write operations in memory tables before flushing sequential sorted runs to secondary disk storage. "
            "Compaction algorithms continuously merge overlapping disk runs in the background to bound read amplification and reclaim obsolete deleted key versions. "
            "This architectural separation optimizes write throughput by converting random disk updates into high-speed sequential streaming writes."
        ),
        # Sub 04 (Colluding Pair 2)
        (
            "High-Throughput Write Architectures: Analysis of LSM-Tree Compaction.\n\n"
            "Log-Structured Merge trees buffer incoming write operations in memory tables before flushing sequential sorted runs to secondary disk storage. "
            "Compaction algorithms continuously merge overlapping disk runs in the background to bound read amplification and reclaim obsolete deleted key versions. "
            "By transforming random disk modifications into sequential log flushes, LSM architectures achieve exceptional write performance in modern NoSQL systems."
        ),
        # Sub 05 (Independent)
        (
            "Spatial Indexing Techniques for Geospatial Information Systems.\n\n"
            "R-tree indexing organizes multidimensional spatial objects into hierarchically nested minimum bounding boxes. "
            "Spatial query processors traverse the bounding box tree to prune non-intersecting regions rapidly during polygon overlap and nearest-neighbor calculations. "
            "Maintaining compact bounding boxes during dynamic insertions prevents overlapping branch degradation in high-density urban geographic information datasets."
        ),
        # Sub 06 (Independent)
        (
            "Bitmap Indexing in Analytical Data Warehousing Environments.\n\n"
            "Bitmap indexes represent attribute values as boolean bit vectors, where each bit corresponds to a distinct table row. "
            "Logical bitwise operators like AND, OR, and NOT execute directly on bit vectors using hardware vector registers with extreme speed. "
            "Bitmap indexing is exceptionally advantageous for low-cardinality attributes like demographic gender or transaction status in read-mostly OLAP workloads."
        ),
        # Sub 07 (Independent)
        (
            "Inverted Indexes in Full-Text Search and Information Retrieval.\n\n"
            "Inverted indexes map individual vocabulary terms to posting lists containing document identifiers and intra-document term positions. "
            "Search engines compress posting lists using variable-byte or delta encoding to minimize disk memory footprints during distributed querying. "
            "Conjunctive queries intersect posting lists sequentially, ranking candidate documents using TF-IDF or BM25 relevance scoring heuristics."
        ),
        # Sub 08 (Independent)
        (
            "Memory-Optimized Indexing for In-Memory Database Systems.\n\n"
            "In-memory database engines dispense with traditional disk page buffer management, designing index structures optimized for processor cache lines. "
            "The Masstree and Adaptive Radix Tree architectures minimize pointer indirection and cache misses through trie-based prefix compression. "
            "Concurrent lock-free read transactions traverse memory indexes without latch contention, maximizing multi-core processor utilization."
        ),
    ]

    collusion_groups = [
        "pair_1", "pair_1",
        "pair_2", "pair_2",
        "none", "none", "none", "none",
    ]

    for i, (c_text, c_grp) in enumerate(zip(CLASS_SUBS, collusion_groups), 1):
        fname = f"class_sub_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(c_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "g_class_submissions",
            "expected_plagiarism": "yes" if c_grp != "none" else "no",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": c_grp,
            "notes": f"Synthetic class submission {i:02d} ({c_grp})",
        })

    # --------------------------------------------------------------------------
    # GROUP H1: 6 Sealed JSON Reports (for seal verify test)
    # --------------------------------------------------------------------------
    for i in range(1, 7):
        fname = f"seal_report_{i:02d}.json"
        sample_text = f"Sample report text for seal verification test case {i}. This document establishes cryptographic seal integrity."
        rep = build_report(sample_text)
        rep_dict = rep.to_dict()
        seal = compute_seal(rep_dict)
        rep_dict["seal"] = seal

        (DOCS_DIR / fname).write_text(json.dumps(rep_dict, indent=2), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "h_seal_reports",
            "expected_plagiarism": "no",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"Cryptographically sealed report {i:02d}",
        })

    # --------------------------------------------------------------------------
    # GROUP H2: 6 Translations in Spanish and French (for Cross-Language test)
    # --------------------------------------------------------------------------
    TRANSLATIONS = [
        # Trans 01: Spanish translation of Ref 01 (Attention)
        (
            "Las arquitecturas modernas de aprendizaje profundo dependen en gran medida de los mecanismos de atencion para tareas de modelado secuencial. "
            "Las redes recurrentes tradicionales procesaban las palabras secuencialmente, creando un cuello de botella informacional para dependencias contextuales distantes. "
            "El mecanismo de autoatencion multicabezal resuelve esta limitacion calculando puntuaciones de interaccion entre todas las posiciones concurrentemente. "
            "Cada cabeza de atencion proyecta vectores de consulta, clave y valor en diferentes subespacios representacionales, permitiendo atender a informacion diversa simultaneamente. "
            "Las codificaciones posicionales se anaden a las representaciones para preservar el orden secuencial de los elementos en el texto procesado."
        ),
        # Trans 02: Spanish translation of Ref 02 (LSH)
        (
            "El hashing sensible a la localidad es una tecnica algoritmica disenada para resolver el problema de vecinos cercanos aproximados en espacios vectoriales. "
            "Los algoritmos tradicionales de hashing intentan minimizar colisiones distribuyendo entradas uniformemente entre las cubetas. "
            "Por el contrario, el hashing sensible a la localidad maximiza deliberadamente las probabilidades de colision para elementos cercanos segun una metrica de distancia. "
            "En el analisis de similitud de conjuntos, las permutaciones MinHash proporcionan un estimador insesgado de la similitud de Jaccard entre colecciones de tokens. "
            "Dividir el vector de firmas en bandas y almacenar cada banda en tablas independientes permite preseleccionar pares de documentos candidatos con complejidad subcuadratica."
        ),
        # Trans 03: Spanish translation of Ref 03 (Raft consensus)
        (
            "Los protocolos de consenso en sistemas distribuidos garantizan que un conjunto de nodos independientes acuerde un estado uniforme de maquina replicada. "
            "Algoritmos clasicos como Paxos garantizan la seguridad bajo condiciones de red asincronas, aunque su complejidad operativa presenta grandes dificultades de implementacion. "
            "El algoritmo de consenso Raft fue formulado especificamente para mejorar la comprensibilidad dividiendo el consenso en problemas independientes: eleccion de lider y replicacion de registros. "
            "Los nodos realizan transiciones entre estados de seguidor, candidato y lider basandose en temporizadores de latidos determinados por el protocolo distribuido."
        ),
        # Trans 04: French translation of Ref 01 (Attention)
        (
            "Les architectures modernes d'apprentissage profond reposent largement sur les mecanismes d'attention pour les taches de modelisation de sequences. "
            "Les reseaux recurrents traditionnels traitaient les jetons d'entree de maniere sequentielle, creant un goulot d'etranglement pour les dependances contextuelles eloignees. "
            "Le mecanisme d'auto-attention multi-tetes surmonte cette limitation en calculant simultanement les scores d'interaction entre toutes les positions de la sequence. "
            "Chaque tete d'attention projette les vecteurs de requete, de cle et de valeur dans des sous-espaces representationnels distincts permettant de capturer des relations variees. "
            "Des encodages positionnels sont ensuite ajoutes aux representations d'entree pour preserver l'ordre sequentiel des elements dans la sequence linguistique."
        ),
        # Trans 05: French translation of Ref 04 (Photosynthesis)
        (
            "La fixation photosynthetique du carbone constitue la voie biochimique par laquelle les organismes autotrophes assimilent le dioxyde de carbone atmospherique en glucides organiques. "
            "Le cycle de Calvin-Benson debute lorsque l'enzyme ribulose-1,5-bisphosphate carboxylase-oxygenase catalyse la carboxylation d'un substrat sucre a cinq carbones. "
            "Cet intermediaire instable se scinde en deux molecules de 3-phosphoglycerate, qui subissent ensuite une reduction enzymatique stimulee par l'ATP et le NADPH issus des reactions photochimiques. "
            "Les trioses phosphates sont ensuite exportes pour la synthese du saccharose ou recycles pour regenerer l'accepteur initial de ribulose."
        ),
        # Trans 06: French translation of Ref 05 (Zero-Knowledge)
        (
            "Les preuves a divulgation nulle de connaissance permettent a un prouveur de convaincre un verificateur sceptique qu'une declaration mathematique est vraie sans reveler d'informations annexes. "
            "Formellement definis par les proprietes de completude, de validite et de confidentialite, ces protocoles interactifs convertissent des relations informatiques en circuits arithmetiques. "
            "Les arguments de connaissance non interactifs utilisent des chaines de reference communes et des couplages sur courbes elliptiques pour comprimer les preuves en engagements cryptographiques concis. "
            "Ces protocoles sont largement appliques dans l'authentification d'identite numerique et la confidentialite sur registres distribues."
        ),
    ]

    for i, t_text in enumerate(TRANSLATIONS, 1):
        fname = f"trans_{i:02d}.txt"
        (DOCS_DIR / fname).write_text(t_text.strip(), encoding="utf-8")
        manifest_rows.append({
            "file": fname,
            "group": "h_translations",
            "expected_plagiarism": "yes",
            "expected_ai": "no",
            "expected_spinner": "no",
            "expected_collusion_group": "none",
            "notes": f"Multilingual translation {i:02d} (ES/FR)",
        })

    # Write manifest CSV
    manifest_path = EVAL_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file",
            "group",
            "expected_plagiarism",
            "expected_ai",
            "expected_spinner",
            "expected_collusion_group",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Generated {len(manifest_rows)} evaluation documents and manifest at {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    build_evaluation_dataset()
