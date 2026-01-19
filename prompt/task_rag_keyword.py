lead_optimization_rag_keywords = ["binding affinity (docking score)", "QED", "SA"]
similarity_rediscovery_rag_keywords = ["similarity"]
boltz_rag_keywords = ["binding affinity (docking score)"]


TASK_RAG_KEYWORD = {
    # Boltz binding affinity prediction
    "boltz/CA2": ["carbonic anhydrase II", "CA2",] + boltz_rag_keywords,
    "boltz/CDK2": ["cyclin-dependent kinase 2", "CDK2"] + boltz_rag_keywords,
    "boltz/DHFR": ["dihydrofolate reductase", "DHFR"] + boltz_rag_keywords,
    "boltz/FABP4": ["fatty acid binding protein 4", "FABP4"] + boltz_rag_keywords,
    "boltz/JNK1": ["c-Jun N-terminal kinase 1", "JNK1"] + boltz_rag_keywords,
    "boltz/P38": ["p38 mitogen-activated protein kinase", "p38 MAPK"] + boltz_rag_keywords,
    "boltz/THROMBIN": ["thrombin", "serine protease"] + boltz_rag_keywords,
    "boltz/TYK2": ["tyrosine kinase 2", "TYK2"] + boltz_rag_keywords,
    # Genmol lead optimization
    "lead_optimization/parp1": ["PARP1"] + lead_optimization_rag_keywords,
    "lead_optimization/fa7": ["FA7"] + lead_optimization_rag_keywords,
    "lead_optimization/5ht1b": ["5-HT1B"] + lead_optimization_rag_keywords,
    "lead_optimization/braf": ["BRAF"] + lead_optimization_rag_keywords,
    "lead_optimization/jak2": ["JAK2"] + lead_optimization_rag_keywords,
    "lead_optimization/sars_cov_2": ["SARS-CoV-2"] + lead_optimization_rag_keywords,
    # MOOD lead optimization
    "lead_optimization_mood/parp1": ["PARP1"] + lead_optimization_rag_keywords,
    "lead_optimization_mood/fa7": ["FA7"] + lead_optimization_rag_keywords,
    "lead_optimization_mood/5ht1b": ["5-HT1B"] + lead_optimization_rag_keywords,
    "lead_optimization_mood/braf": ["BRAF"] + lead_optimization_rag_keywords,
    "lead_optimization_mood/jak2": ["JAK2"] + lead_optimization_rag_keywords,
    "lead_optimization_mood/sars_cov_2": ["SARS-CoV-2"] + lead_optimization_rag_keywords,
    # PMO
    "pmo/Amlodipine_MPO": ["amlodipine"],
    "pmo/Celecoxib_Rediscovery": ["celecoxib"] + similarity_rediscovery_rag_keywords,
    "pmo/DRD2": ["DRD2 receptor (Dopamine Receptor D2) binding affinitiy"],
    "pmo/Fexofenadine_MPO": ["fexofenadine", "TPSA", "LogP"],
    "pmo/JNK3": ["JNK3 (c-Jun N-terminal kinase 3) inhibitory activity"],
    "pmo/GSK3B": ["GSK3B (Glycogen Synthase Kinase 3 Beta) inhibitory activity"],
    "pmo/Median1": ["Camphor", "Menthol"],
    "pmo/Median2": ["Tadalafil", "Sildenafil"],
    "pmo/Osimertinib_MPO": ["osimertinib", "TPSA", "LogP"],
    "pmo/Perindopril_MPO": ["perindopril", "TPSA", "LogP"],
    "pmo/Ranolazine_MPO": ["ranolazine", "TPSA", "LogP"],
    "pmo/Sitagliptin_MPO": ["sitagliptin", "TPSA", "LogP"],
    "pmo/Zaleplon_MPO": ["zaleplon"],
    "pmo/Albuterol_similarity": ["albuterol"] + similarity_rediscovery_rag_keywords,
    "pmo/Valsartan_smarts": ["valsartan", "logP", "TPSA", "Bertz complexity"],
    "pmo/Thiothixene_Rediscovery": ["thiothixene"] + similarity_rediscovery_rag_keywords,
    "pmo/Troglitazone_Rediscovery": ["troglitazone"] + similarity_rediscovery_rag_keywords,
    "pmo/QED": ["QED"],
    "pmo/Mestranol_similarity": ["mestranol"] + similarity_rediscovery_rag_keywords,
    "pmo/Scaffold_hop": ["scaffold hop"],
    "pmo/Deco_hop": ["deco hop"],
    "pmo/Isomers_c7h8n2o2": ["isomers c7h8n2o2"],
    "pmo/Isomers_c9h10n2o2pf2cl": ["isomers c9h10n2o2pf2cl"],
}