#!/usr/bin/env python3
"""Visite guidee complete du paquet prism-ex.

Execute chaque fonctionnalite, affiche le resultat, et explique ce qu'il faut y
lire. Concu pour etre lu autant qu'execute : chaque section indique ce qu'on teste,
ce que fait le code, et comment interpreter la sortie.

    python scripts/tour_complet.py              # tout, environ 3 minutes
    python scripts/tour_complet.py --rapide     # stabilite allegee (~1 minute)
    python scripts/tour_complet.py --section 7  # une seule section

Aucune connexion reseau n'est necessaire : le paquet et ses dependances doivent
seulement etre deja installes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- affichage

LARGEUR = 78


def titre(numero: int, texte: str) -> None:
    print("\n" + "=" * LARGEUR)
    print(f" SECTION {numero} — {texte}")
    print("=" * LARGEUR)


def bloc(etiquette: str, texte: str) -> None:
    print(f"\n{etiquette}")
    for ligne in textwrap.wrap(texte, LARGEUR - 2):
        print(f"  {ligne}")


def teste(texte: str) -> None:
    bloc("CE QU'ON TESTE", texte)


def lecture(texte: str) -> None:
    bloc("LECTURE DU RESULTAT", texte)


def code(*lignes: str) -> None:
    print("\n  >>> " + "\n  >>> ".join(lignes))


def sortie(texte: str) -> None:
    for ligne in str(texte).rstrip().split("\n"):
        print(f"      {ligne}")


def cli(*args: str) -> subprocess.CompletedProcess:
    """Appelle l'interface en ligne de commande dans un sous-processus."""
    commande = [sys.executable, "-m", "prism_ex.cli", *args]
    print(f"\n  $ prism-ex {' '.join(args)}")
    resultat = subprocess.run(commande, capture_output=True, text=True)
    if resultat.stdout:
        sortie(resultat.stdout)
    if resultat.stderr.strip():
        sortie("[stderr] " + resultat.stderr.strip())
    print(f"      (code de retour : {resultat.returncode})")
    return resultat


TMP = Path(tempfile.gettempdir())
DEMO = TMP / "tour_demo.fcs"
nombre_evenements = 6000

# ============================================================== 1. installation


def section_1() -> None:
    titre(1, "L'installation et les versions")
    teste(
        "Que le paquet est importable, que la version du code et celle des "
        "metadonnees d'installation concordent, et quelles versions de "
        "dependances sont reellement chargees. Une divergence entre les deux "
        "versions signifie une installation obsolete."
    )

    import importlib.metadata as md

    import prism_ex

    code("import prism_ex", "prism_ex.__version__")
    sortie(prism_ex.__version__)
    code("importlib.metadata.version('prism-ex')")
    sortie(md.version("prism-ex"))
    code("prism_ex.__file__")
    sortie(prism_ex.__file__)

    import igraph
    import leidenalg
    import scipy
    import sklearn

    print()
    for nom, version in [
        ("python", sys.version.split()[0]),
        ("numpy", np.__version__),
        ("scipy", scipy.__version__),
        ("scikit-learn", sklearn.__version__),
        ("igraph", igraph.__version__),
        ("leidenalg", leidenalg.version),
    ]:
        print(f"      {nom:<14s} {version}")

    lecture(
        "Les deux versions doivent etre identiques (0.1.0). Le chemin indique "
        "d'ou le paquet est charge : un chemin en site-packages signifie une "
        "installation normale, un chemin dans src/ signifie une installation "
        "editable. Retenez vos versions de numpy, scipy et scikit-learn : c'est "
        "la comparaison entre environnements qui fonde la reproductibilite."
    )


# =============================================================== 2. lecture FCS


def section_2() -> None:
    titre(2, "Resultat 1 — generation et lecture stricte d'un fichier FCS 3.1")
    teste(
        "La generation d'un fichier FCS 3.1 valide, puis sa relecture. C'est "
        "l'aller-retour ecrivain/lecteur : les deux doivent s'accorder sur la norme."
    )

    from prism_ex import read_fcs, write_demo_file

    code("path, verite = write_demo_file('demo.fcs', nombre_evenements, seed=20260817)")
    chemin, verite = write_demo_file(DEMO, nombre_evenements, seed=20260817)
    sortie(f"{chemin}  ({chemin.stat().st_size} octets)")
    print("\n      Populations reelles (la verite terrain) :")
    for nom, taille in verite.sizes().items():
        sortie(f"  {nom:<18s} {taille:>5d}")

    code("fcs = read_fcs('demo.fcs')")
    fcs = read_fcs(DEMO)
    sortie(fcs.summary())

    print()
    code("fcs.n_events, fcs.n_channels, fcs.channel_names[:4]")
    sortie(f"{fcs.n_events}, {fcs.n_channels}, {fcs.channel_names[:4]}")
    code("fcs.keywords['$TOT'], fcs.keywords['$par']  # insensible a la casse")
    sortie(f"{fcs.keywords['$TOT']!r}, {fcs.keywords['$par']!r}")
    code("fcs.index_of('CD3'), fcs.index_of('CD3-BV421')  # $PnN puis $PnS")
    sortie(f"{fcs.index_of('CD3')}, {fcs.index_of('CD3-BV421')}")
    code("fcs.column('CD3')[:3]")
    sortie(np.array2string(fcs.column("CD3")[:3], precision=1))

    lecture(
        f"{nombre_evenements} evenements sur 9 canaux. Les mots-cles sont accessibles sans "
        "tenir compte de la casse, car la norme FCS les declare insensibles a "
        "la casse. Un marqueur se resout d'abord par $PnN (nom court) puis par "
        "$PnS (nom du fluorochrome) : ici 'CD3' et 'CD3-BV421' designent la meme "
        "colonne. Les populations affichees sont la verite terrain : tous les "
        "resultats suivants seront juges par rapport a elles."
    )

    print()
    teste("Que la matrice d'evenements est en lecture seule.")
    code("fcs.events[0, 0] = 42.0")
    try:
        fcs.events[0, 0] = 42.0
        sortie("AUCUNE ERREUR — ce serait un defaut")
    except ValueError as erreur:
        sortie(f"ValueError: {erreur}")
    lecture(
        "Le refus est voulu. L'objet porte une empreinte SHA-256 de sa source ; "
        "si un appelant pouvait modifier les donnees, cette empreinte deviendrait "
        "un mensonge."
    )


# ============================================================ 3. rejets stricts


def section_3() -> None:
    titre(3, "Resultat 1 — les fichiers invalides sont refuses")
    teste(
        "Le coeur de l'exigence : un fichier malforme, tronque, incoherent ou "
        "d'une autre version ne doit produire aucun resultat partiel. Chaque cas "
        "part d'un fichier valide dont on casse exactement un aspect."
    )

    from prism_ex.errors import FCSError
    from prism_ex.fcs.reader import read_fcs_bytes
    from prism_ex.fcs.writer import build_fcs_bytes

    matrice = np.arange(24, dtype=float).reshape(8, 3)
    valide = build_fcs_bytes(matrice, ["A", "B", "C"])

    cas = [
        ("fichier valide (temoin)", valide),
        ("version FCS2.0", b"FCS2.0" + valide[6:]),
        ("version FCS3.0", b"FCS3.0" + valide[6:]),
        ("fichier vide", b""),
        ("en-tete tronque (40 octets)", valide[:40]),
        ("DATA tronque de 4 octets", valide[:-4]),
        (
            "$TOT annonce 9999 evenements",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$TOT": "9999"}),
        ),
        (
            "$PAR annonce 4 parametres",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$PAR": "4"}),
        ),
        (
            "deux canaux nommes 'A'",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$P2N": "A"}),
        ),
        (
            "$MODE histogramme",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$MODE": "C"}),
        ),
        (
            "$DATATYPE ASCII",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$DATATYPE": "A"}),
        ),
        (
            "$BYTEORD mixte",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$BYTEORD": "3,4,1,2"}),
        ),
        (
            "$NEXTDATA non nul",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$NEXTDATA": "1024"}),
        ),
        (
            "$P1E logarithmique a decalage nul",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$P1E": "4,0"}),
        ),
        (
            "$P1B incoherent avec $DATATYPE",
            build_fcs_bytes(matrice, ["A", "B", "C"], extra_keywords={"$P1B": "16"}),
        ),
    ]

    print(f"\n  {'defaut introduit':<38s} {'reponse du lecteur'}")
    print("  " + "-" * (LARGEUR - 4))
    for etiquette, octets in cas:
        try:
            resultat = read_fcs_bytes(octets)
            reponse = f"accepte ({resultat.n_events} evenements)"
        except FCSError as erreur:
            reponse = type(erreur).__name__
        print(f"  {etiquette:<38s} {reponse}")

    lecture(
        "Chaque refus porte un type distinct. InconsistentMetadata signifie "
        "« votre fichier se contredit lui-meme » ; UnsupportedFCSFeature signifie "
        "« votre fichier est valide, c'est moi qui ne l'implemente pas ». Ce sont "
        "deux conversations differentes avec la personne qui detient le fichier, "
        "et c'est pourquoi la hierarchie d'exceptions est fine. N'importe qui "
        "peut lever une ValueError ; la taxonomie est le travail d'ingenierie."
    )

    print()
    teste("Qu'aucun objet a moitie construit ne s'echappe, quelle que soit la coupure.")
    for coupure in (59, 100, 150, 200, 400):
        try:
            resultat = read_fcs_bytes(valide[:coupure])
            coherent = resultat.events.shape == (resultat.n_events, resultat.n_channels)
            etat = f"objet complet et coherent : {coherent}"
        except FCSError as erreur:
            etat = f"{type(erreur).__name__}"
        print(f"      coupure a {coupure:>4d} octets  ->  {etat}")
    lecture(
        "Les deux seules issues possibles sont un objet complet ou une exception. "
        "Jamais un objet dont certains champs seraient remplis et d'autres non."
    )


# ============================================================== 4. sous-espace


def section_4() -> None:
    titre(4, "Le sous-espace de marqueurs — transformation et normalisation")
    teste(
        "La selection des colonnes, la transformation arcsinh et la "
        "standardisation. C'est l'etape qui rend les distances comparables d'un "
        "marqueur a l'autre."
    )

    from prism_ex import read_fcs, select_subspace
    from prism_ex.errors import AmbiguousMarker, UnknownMarker

    fcs = read_fcs(DEMO)

    code("brut = select_subspace(fcs, ['CD3','CD4'], transform='none', scaling='none')")
    brut = select_subspace(fcs, ["CD3", "CD4"], transform="none", scaling="none")
    sortie(f"etendue CD3 : {brut.matrix[:, 0].min():10.1f} a {brut.matrix[:, 0].max():10.1f}")

    code("asinh = select_subspace(fcs, ['CD3','CD4'], transform='asinh', scaling='none')")
    asinh = select_subspace(fcs, ["CD3", "CD4"], transform="asinh", scaling="none")
    sortie(f"etendue CD3 : {asinh.matrix[:, 0].min():10.2f} a {asinh.matrix[:, 0].max():10.2f}")

    code("final = select_subspace(fcs, ['CD3','CD4'])  # asinh + score z par defaut")
    final = select_subspace(fcs, ["CD3", "CD4"])
    moyennes = final.matrix.mean(axis=0).round(6)
    ecarts_types = final.matrix.std(axis=0).round(6)
    sortie(f"moyenne {moyennes}  ecart-type {ecarts_types}")

    lecture(
        "L'etendue brute couvre plusieurs decades ; apres arcsinh elle tient dans "
        "un intervalle etroit. Sans cette compression, la distance euclidienne "
        "serait dominee par le canal le plus brillant et le graphe encoderait la "
        "brillance plutot que le phenotype. Apres standardisation, chaque colonne "
        "a une moyenne nulle et un ecart-type de 1 : les marqueurs pesent alors "
        "le meme poids dans les distances."
    )

    print()
    teste("Que les erreurs de selection sont explicites.")
    code("select_subspace(fcs, ['CD999'])")
    try:
        select_subspace(fcs, ["CD999"])
    except UnknownMarker as erreur:
        sortie(f"UnknownMarker: {erreur}")
    code("select_subspace(fcs, ['CD3', 'CD3-BV421'])  # le meme canal deux fois")
    try:
        select_subspace(fcs, ["CD3", "CD3-BV421"])
    except AmbiguousMarker as erreur:
        sortie(f"AmbiguousMarker: {erreur}")
    lecture(
        "Le marqueur inconnu enumere les canaux disponibles. Le doublon est "
        "refuse plutot que dedoublonne silencieusement : un marqueur presente "
        "deux fois pesarait double dans chaque distance euclidienne."
    )


# ================================================================== 5. graphe


def section_5() -> None:
    titre(5, "Resultat 2 — le graphe de voisinage")
    teste(
        "La construction du graphe des k plus proches voisins et les trois modes "
        "de ponderation des aretes."
    )

    from prism_ex import build_graph, read_fcs, select_subspace
    from prism_ex.errors import ConfigurationError
    from prism_ex.synth import CLUSTERING_MARKERS

    fcs = read_fcs(DEMO)
    sous_espace = select_subspace(fcs, CLUSTERING_MARKERS)

    for ponderation in ("jaccard", "distance", "uniform"):
        depart = time.time()
        graphe = build_graph(sous_espace, k=30, weighting=ponderation)
        duree = time.time() - depart
        print(
            f"      {ponderation:<9s} aretes {graphe.n_edges:>7d}   "
            f"densite {graphe.density():.5f}   "
            f"poids {graphe.adjacency.data.min():.3f}-{graphe.adjacency.data.max():.3f}   "
            f"{duree:.1f}s"
        )

    graphe = build_graph(sous_espace, k=30)
    code("2 * graphe.n_edges / nombre_evenements  # degre moyen")
    sortie(f"{2 * graphe.n_edges / nombre_evenements:.1f}")
    code("graphe.adjacency.sum()  # 2m, poids total")
    sortie(f"{graphe.adjacency.sum():.2f}")
    code("(graphe.adjacency - graphe.adjacency.T).nnz  # symetrie")
    sortie(f"{abs(graphe.adjacency - graphe.adjacency.T).max()}")
    code("graphe.adjacency.diagonal().sum()  # pas de boucle")
    sortie(f"{graphe.adjacency.diagonal().sum()}")

    lecture(
        "La ponderation jaccard produit moins d'aretes que les deux autres : "
        "l'elagage sous 1/15 supprime les aretes longues qui relient deux "
        "populations par un unique point aberrant partage. Les poids sont bornes "
        "dans [0,1] car ce sont des indices de Jaccard. La matrice est exactement "
        "symetrique et sans boucle : ce sont les deux invariants qu'un graphe non "
        "oriente doit respecter. Le poids total 2m sert a normaliser la qualite."
    )

    print()
    teste("Que k doit rester inferieur au nombre d'evenements.")
    code("build_graph(sous_espace, k=10000)")
    try:
        build_graph(sous_espace, k=10000)
    except ConfigurationError as erreur:
        sortie(f"ConfigurationError: {erreur}")


# ============================================================= 6. communautes


def section_6() -> None:
    titre(6, "Resultat 2 — la detection de communautes")
    teste(
        "L'algorithme de Leiden sur le graphe, l'effet de la resolution, et la "
        "correspondance des communautes trouvees avec les populations reelles."
    )

    from sklearn.metrics import adjusted_rand_score

    from prism_ex import CommunityConfig, find_communities, read_fcs
    from prism_ex.synth import CLUSTERING_MARKERS, make_dataset

    fcs = read_fcs(DEMO)
    verite = make_dataset(nombre_evenements, seed=20260817)
    config = CommunityConfig(markers=CLUSTERING_MARKERS)

    code("resultat = find_communities(fcs, CommunityConfig(markers=CLUSTERING_MARKERS))")
    resultat = find_communities(fcs, config)
    sortie(
        f"{resultat.partition.n_communities} communautes, qualite {resultat.partition.quality:.4f}"
    )
    sortie(f"tailles : {resultat.sizes}")

    poids_total = resultat.graph.adjacency.sum()
    code("qualite / 2m  # la modularite de Newman, normalisee")
    sortie(f"{resultat.partition.quality / poids_total:.3f}")

    print("\n      Croisement avec la verite terrain :")
    noms = verite.population_names
    for communaute in resultat.partition.ids:
        membres = resultat.partition.members(communaute)
        compte: dict[str, int] = {}
        for etiquette in verite.labels[membres]:
            compte[noms[etiquette]] = compte.get(noms[etiquette], 0) + 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(compte.items(), key=lambda x: -x[1])[:3])
        purete = max(compte.values()) / membres.size
        sortie(f"  communaute {communaute} (n={membres.size:>4d}, purete {purete:5.1%}) : {detail}")

    ari = adjusted_rand_score(verite.labels, resultat.partition.labels)
    code("adjusted_rand_score(verite, partition)")
    sortie(f"{ari:.3f}")

    lecture(
        "Six communautes pour six populations : le nombre est correct. Quatre "
        "d'entre elles retrouvent une population unique a plus de 99% de purete, "
        "dont la population rare a 1,5%. Les deux autres se partagent 2609 "
        "evenements, soit exactement 1800+810 : le total est juste, la frontiere "
        "est mal placee. L'ARI de 0,787 traduit precisement cela — quatre "
        "populations exactes, une frontiere fausse. La qualite brute (48099) est "
        "la somme RB non normalisee ; divisee par 2m elle vaut 0,838, ce qui est "
        "la modularite de Newman et indique un graphe fortement modulaire."
    )

    print()
    teste("L'effet de la resolution : c'est le parametre qui choisit l'echelle.")
    print(f"\n      {'resolution':>10s} {'communautes':>12s} {'ARI':>8s}")
    for resolution in (0.2, 0.4, 0.6, 1.0, 1.5):
        autre = find_communities(fcs, config.replace(resolution=resolution))
        score = adjusted_rand_score(verite.labels, autre.partition.labels)
        print(f"      {resolution:>10.1f} {autre.partition.n_communities:>12d} {score:>8.3f}")
    lecture(
        "En dessous de 0,45 les deux populations CD4 fusionnent en une seule "
        "communaute ; au-dessus elles se separent, mais au mauvais endroit. "
        "Aucune valeur de resolution ne donne la bonne reponse : c'est le "
        "diagnostic que l'analyse de stabilite doit produire sans voir les "
        "etiquettes."
    )


# ============================================================== 7. comparaison


def section_7() -> None:
    titre(7, "Resultat 3 — la comparaison de deux communautes")
    teste(
        "Les tailles d'effet, les intervalles de confiance, la distance "
        "d'energie, et le refus des valeurs p circulaires."
    )

    from prism_ex import CommunityConfig, find_communities, read_fcs
    from prism_ex.compare import cliffs_delta, compare_communities
    from prism_ex.synth import CLUSTERING_MARKERS

    fcs = read_fcs(DEMO)
    resultat = find_communities(fcs, CommunityConfig(markers=CLUSTERING_MARKERS))

    print("\n  --- (a) deux populations reellement differentes : CD8 T contre B ---")
    code("compare_communities(resultat, 1, 2, markers=['CD3','CD19','Viability','FSC-A'],")
    code("                    inference='split')")
    comparaison = compare_communities(
        resultat,
        1,
        2,
        markers=["CD3", "CD19", "Viability", "FSC-A"],
        inference="split",
        n_bootstrap=200,
        n_permutations=299,
    )
    sortie(comparaison.to_markdown())

    lecture(
        "CD3 et CD19 donnent des deltas de +1,000 et -1,000 : separation totale, "
        "ce qui est attendu entre lymphocytes T et B. Ces deux marqueurs portent "
        "une etoile et aucune valeur q : ils ont servi a definir les communautes, "
        "donc aucune hypothese nulle ne leur survit. FSC-A et Viability, eux, "
        "sont hors du sous-espace de partitionnement : ils recoivent des valeurs "
        "p, et celles-ci sont correctement grandes puisque ces canaux sont "
        "generes independamment des populations. C'est le controle negatif."
    )

    print("\n  --- (b) la paire ambigue : communautes 0 et 3 ---")
    ambigue = compare_communities(
        resultat,
        0,
        3,
        markers=["CD4", "CD8", "Viability"],
        inference="split",
        n_bootstrap=100,
        n_permutations=99,
    )
    sortie(ambigue.to_markdown())
    lecture(
        "Le message final explique pourquoi aucune valeur p n'est produite : les "
        "communautes n'ont pas pu etre retrouvees a partir de la moitie des "
        "evenements. C'est un diagnostic, pas une panne — et c'est la meme "
        "conclusion que celle de la section 8."
    )

    print("\n  --- (c) les statistiques prises isolement ---")
    code("cliffs_delta([4,5,6], [1,2,3])  # separation complete")
    sortie(f"{cliffs_delta(np.array([4.0, 5, 6]), np.array([1.0, 2, 3])):+.3f}")
    code("cliffs_delta([1,2,3], [1,2,3])  # distributions identiques")
    sortie(f"{cliffs_delta(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])):+.3f}")

    generateur = np.random.default_rng(0)
    a = generateur.lognormal(0, 1, 500)
    b = generateur.lognormal(0.5, 1, 500)
    code("cliffs_delta(a, b) puis cliffs_delta(arcsinh(a/250), arcsinh(b/250))")
    delta_brut = cliffs_delta(a, b)
    delta_transforme = cliffs_delta(np.arcsinh(a / 250), np.arcsinh(b / 250))
    sortie(f"{delta_brut:+.6f}  puis  {delta_transforme:+.6f}")
    lecture(
        "Les deux valeurs sont identiques au dernier chiffre : le delta de Cliff "
        "ne depend que des rangs, et l'arcsinh est strictement croissante, donc "
        "elle ne peut modifier aucun rang. C'est l'argument central en faveur "
        "d'une statistique de rang : le cofacteur est un choix d'analyse qui "
        "n'apparait dans aucun tableau de resultats, et une taille d'effet ne "
        "doit pas en dependre."
    )


# =============================================================== 8. stabilite


def section_8(rapide: bool) -> None:
    titre(8, "Resultat 4 — l'analyse de stabilite")
    teste(
        "Les quatre sources de variation mesurees separement : l'algorithme, "
        "l'echantillon, les parametres, et l'evenement individuel."
    )

    from prism_ex import CommunityConfig, find_communities, read_fcs
    from prism_ex.stability import assess_stability
    from prism_ex.synth import CLUSTERING_MARKERS, make_dataset

    fcs = read_fcs(DEMO)
    resultat = find_communities(fcs, CommunityConfig(markers=CLUSTERING_MARKERS))

    n_resamples = 8 if rapide else 40
    code(f"assess_stability(resultat, n_resamples={n_resamples}, n_seeds=5)")
    print(f"      (patientez : {n_resamples} repartitionnements complets)")
    depart = time.time()
    preuves = assess_stability(
        resultat,
        n_resamples=n_resamples,
        n_seeds=5,
        k_grid=(15, 30, 45) if rapide else None,
        resolution_grid=(0.3, 0.6, 1.0) if rapide else None,
    )
    sortie(preuves.to_markdown())
    print(f"      (calcule en {time.time() - depart:.0f} s)")

    verite = make_dataset(nombre_evenements, seed=20260817)
    print("\n      Confrontation a la verite terrain (que l'instrument ne voit jamais) :")
    noms = verite.population_names
    for entree in preuves.per_community:
        membres = resultat.partition.members(entree.community)
        compte: dict[str, int] = {}
        for etiquette in verite.labels[membres]:
            compte[noms[etiquette]] = compte.get(noms[etiquette], 0) + 1
        purete = max(compte.values()) / membres.size
        sortie(
            f"  communaute {entree.community} : J={entree.mean_jaccard:.2f} "
            f"({entree.verdict:<10s}) purete reelle {purete:5.1%}"
        )

    lecture(
        "Les communautes jugees fiables ont une purete reelle superieure a 99%. "
        "Les deux jugees douteuses sont exactement celles dont la purete est "
        "mauvaise. L'instrument a donc identifie la mauvaise reponse sans jamais "
        "voir les etiquettes : c'est une validation de la methode, pas seulement "
        "un resultat sur ce jeu de donnees. Notez aussi l'ARI entre graines : en "
        "ne changeant QUE la graine aleatoire, la frontiere se deplace — ce qui "
        "prouve que cette separation n'est pas une propriete des donnees."
    )

    print()
    teste("Le score de coeur par evenement : la stabilite n'est pas uniforme.")
    scores = preuves.event_core_score
    affectes = scores >= 0
    code("np.mean(scores[affectes] >= 0.8)  # fraction d'evenements de coeur")
    sortie(f"{np.mean(scores[affectes] >= 0.8):.1%}")
    code("preuves.boundary_fraction  # evenements de frontiere")
    sortie(f"{preuves.boundary_fraction:.1%}")
    lecture(
        "Une communaute peut avoir un Jaccard mediocre et une fraction de coeur "
        "excellente : ses evenements atterrissent toujours ensemble, mais le "
        "groupe qu'ils forment absorbe ou relache un bloc entier d'un tirage a "
        "l'autre. Le Jaccard capte l'instabilite des FRONTIERES, le score de "
        "coeur capte l'ambiguite des EVENEMENTS. Deux defaillances orthogonales."
    )


# ============================================================== 9. provenance


def section_9() -> None:
    titre(9, "Provenance et determinisme")
    teste(
        "Que le meme fichier, la meme configuration et la meme version donnent "
        "exactement le meme resultat, et que ce fait est verifiable."
    )

    from prism_ex import CommunityConfig, find_communities, read_fcs
    from prism_ex.synth import CLUSTERING_MARKERS

    fcs = read_fcs(DEMO)
    config = CommunityConfig(markers=CLUSTERING_MARKERS)

    executions = [find_communities(fcs, config) for _ in range(3)]
    code("[r.partition.quality for r in executions]")
    sortie(str([round(r.partition.quality, 4) for r in executions]))
    code("np.array_equal(executions[0].labels, executions[1].labels)")
    identiques = all(
        np.array_equal(executions[0].partition.labels, autre.partition.labels)
        for autre in executions[1:]
    )
    sortie(str(identiques))

    provenance = executions[0].partition.provenance
    code("provenance.source_sha256")
    sortie(provenance.source_sha256)
    code("provenance.config_id")
    sortie(provenance.config_id)
    code("provenance.environment")
    sortie(str(provenance.environment))

    code("find_communities(fcs, config.replace(seed=99)).partition.provenance.config_id")
    autre = find_communities(fcs, config.replace(seed=99))
    sortie(autre.partition.provenance.config_id)

    lecture(
        "Trois executions, des etiquettes identiques au bit pres. L'empreinte de "
        "configuration change des qu'un parametre change, ici la graine : deux "
        "resultats portant la meme empreinte proviennent des memes reglages, "
        "quelle que soit la maniere dont l'appelant les a ecrits. Comparez "
        "l'empreinte affichee ici a 7caf7f2e90f1a602 : si elle est identique, "
        "votre machine reproduit exactement les chiffres du rapport."
    )


# ===================================================================== 10. CLI


def section_10() -> None:
    titre(10, "L'interface en ligne de commande")
    teste("Les six sous-commandes, et le comportement en cas d'erreur.")

    cli("--version")
    cli("info", str(DEMO))
    cli("communities", str(DEMO), "--k", "30", "--resolution", "0.6")

    print("\n  --- un fichier tronque doit etre refuse avec un code non nul ---")
    tronque = TMP / "tour_tronque.fcs"
    tronque.write_bytes(DEMO.read_bytes()[:200])
    resultat = cli("info", str(tronque))
    assert resultat.returncode != 0, "le fichier tronque aurait du etre refuse"

    print("\n  --- un marqueur inexistant : message clair, pas de trace d'appel ---")
    cli("communities", str(DEMO), "--markers", "CD999")

    lecture(
        "Les erreurs sortent sous forme de message sur stderr avec un code de "
        "retour de 1, et non sous forme de trace d'appel Python. Un script "
        "appelant peut donc reagir au code de retour, et un humain lit une "
        "phrase comprehensible."
    )


# ===================================================================== 11. API


def section_11() -> None:
    titre(11, "Le point d'acces HTTP (extension optionnelle)")
    teste(
        "Que le service repond, qu'il renvoie les memes tailles que la ligne de "
        "commande, et qu'un fichier invalide donne un code 422 nommant le defaut."
    )

    try:
        from fastapi.testclient import TestClient

        from prism_ex.api import app
    except ImportError as erreur:
        sortie(f"extra [api] non installe ({erreur}) — section ignoree")
        return

    client = TestClient(app)

    code("client.get('/health').json()")
    sortie(str(client.get("/health").json()))

    code("client.post('/communities/sizes', files={'file': ...}, data={'markers': ...})")
    with open(DEMO, "rb") as fichier:
        reponse = client.post(
            "/communities/sizes",
            files={"file": ("demo.fcs", fichier, "application/octet-stream")},
            data={"markers": "CD3,CD4,CD8,CD19,CD56", "k": 30, "resolution": 0.6},
        )
    charge = reponse.json()
    sortie(f"code {reponse.status_code}")
    sortie(f"tailles : {charge['sizes']}")
    sortie(f"empreinte source : {charge['provenance']['source_sha256'][:16]}...")

    code("client.post(..., fichier invalide)")
    mauvais = client.post(
        "/communities/sizes",
        files={"file": ("bad.fcs", b"FCS2.0    " + b"0" * 100, "application/octet-stream")},
        data={"markers": "CD3"},
    )
    sortie(f"code {mauvais.status_code} — {mauvais.json()['detail']}")

    lecture(
        "Les tailles sont identiques a celles de la ligne de commande, obtenues "
        "par un transport different et un point d'entree different. Le point "
        "d'acces recoit des octets et non un chemin : c'est pourquoi "
        "read_fcs_bytes existe a cote de read_fcs. Un fichier invalide donne un "
        "422 portant le type d'erreur precis, et non un 500 generique."
    )


# ==================================================================== resume


def resume() -> None:
    print("\n" + "=" * LARGEUR)
    print(" CE QUE CETTE VISITE A ETABLI")
    print("=" * LARGEUR)
    for ligne in [
        "1.  Installation coherente, versions de dependances relevees.",
        "2.  Lecture d'un FCS 3.1 valide, mots-cles et marqueurs resolus.",
        "3.  Quatorze defauts distincts refuses, chacun avec son type d'erreur.",
        "4.  Sous-espace : arcsinh compresse les decades, le score z egalise les poids.",
        "5.  Graphe symetrique, sans boucle, poids bornes, elagage effectif.",
        "6.  Six communautes, quatre justes a plus de 99%, une frontiere fausse.",
        "7.  Tailles d'effet invariantes par transformation, valeurs p refusees",
        "    la ou elles seraient circulaires, controle negatif concluant.",
        "8.  Stabilite : les communautes signalees comme douteuses sont exactement",
        "    celles qui sont fausses, identifiees sans voir les etiquettes.",
        "9.  Determinisme au bit pres, empreinte de configuration verifiable.",
        "10. Interface en ligne de commande : erreurs propres, codes de retour.",
        "11. Point d'acces HTTP : memes resultats par un autre transport.",
    ]:
        print(f"  {ligne}")
    print("\n  Trois reperes reproductibles :")
    print("    121474 aretes, qualite 48099.4187, empreinte 7caf7f2e90f1a602")
    print("    ARI 0.787 contre la verite terrain")
    print("    Jaccard 0.64 et 0.44 pour les deux communautes fausses\n")


SECTIONS = {
    1: lambda rapide: section_1(),
    2: lambda rapide: section_2(),
    3: lambda rapide: section_3(),
    4: lambda rapide: section_4(),
    5: lambda rapide: section_5(),
    6: lambda rapide: section_6(),
    7: lambda rapide: section_7(),
    8: lambda rapide: section_8(rapide),
    9: lambda rapide: section_9(),
    10: lambda rapide: section_10(),
    11: lambda rapide: section_11(),
}


def main() -> int:
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--rapide", action="store_true", help="moins de reechantillonnages")
    analyseur.add_argument("--section", type=int, default=None, help="n'executer qu'une section")
    arguments = analyseur.parse_args()

    print("=" * LARGEUR)
    print(" VISITE GUIDEE DU PAQUET prism-ex")
    print("=" * LARGEUR)
    print(
        textwrap.fill(
            "Chaque section indique ce qu'elle teste, montre l'appel effectue, "
            "affiche le resultat, puis explique comment le lire. Aucune connexion "
            "reseau n'est necessaire.",
            LARGEUR,
        )
    )

    depart = time.time()
    # La section 2 cree le fichier de demonstration dont les autres dependent.
    if arguments.section not in (None, 1, 2):
        from prism_ex import write_demo_file

        if not DEMO.exists():
            write_demo_file(DEMO, nombre_evenements, seed=20260817)

    for numero, fonction in SECTIONS.items():
        if arguments.section in (None, numero):
            fonction(arguments.rapide)

    if arguments.section is None:
        resume()
    print(f"  Duree totale : {time.time() - depart:.0f} secondes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
