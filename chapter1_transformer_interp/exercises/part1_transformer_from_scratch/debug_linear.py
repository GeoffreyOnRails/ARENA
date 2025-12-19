"""
Script de débogage pour reproduire exactement le calcul de nn.Linear
et identifier la cause de l'écart entre CPU et MPS.

Ce script reproduit le calcul exact tel qu'exécuté par nn.Linear en utilisant
les mêmes opérations de bas niveau (addmm, F.linear, etc.)

Usage:
    - Exécution directe: python debug_linear.py
    - Import dans notebook: from debug_linear import debug_linear_computation
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformer_lens import HookedTransformer

# ============================================================================
# DÉFINITIONS DES FONCTIONS DE REPRODUCTION
# ============================================================================

def reproduce_with_nn_linear(z_input, W_O, b_O, device):
    """
    Reproduit le calcul en utilisant directement nn.Linear.
    C'est la méthode de référence qui reproduit exactement le bug.
    """
    batch, pos, n_heads, d_head = z_input.shape
    d_model = W_O.shape[-1]
    d_in = n_heads * d_head
    
    # Aplatir z: [batch, pos, n_heads, d_head] -> [batch*pos, n_heads*d_head]
    z_flat = z_input.flatten(start_dim=0, end_dim=1)  # [batch*pos, n_heads, d_head]
    z_flat = z_flat.flatten(start_dim=1)  # [batch*pos, n_heads*d_head]
    
    # Aplatir W_O: [n_heads, d_head, d_model] -> [n_heads*d_head, d_model]
    W_O_flat = W_O.flatten(start_dim=0, end_dim=1)  # [n_heads*d_head, d_model]
    
    # Créer un module nn.Linear
    linear_module = nn.Linear(in_features=d_in, out_features=d_model, bias=True).to(device)
    linear_module.weight.data = W_O_flat.T  # nn.Linear attend [out_features, in_features]
    linear_module.bias.data = b_O
    
    # Exécuter
    output_flat = linear_module(z_flat)  # [batch*pos, d_model]
    
    # Reshape: [batch*pos, d_model] -> [batch, pos, d_model]
    output = output_flat.reshape(batch, pos, -1)
    return output


def reproduce_with_F_linear(z_input, W_O, b_O, device):
    """
    Reproduit exactement le calcul de nn.Linear en utilisant F.linear.
    F.linear fait: output = input @ weight.T + bias
    """
    # Aplatir z: [batch, pos, n_heads, d_head] -> [batch*pos, n_heads*d_head]
    batch, pos, n_heads, d_head = z_input.shape
    z_flat = z_input.flatten(start_dim=0, end_dim=1)  # [batch*pos, n_heads, d_head]
    z_flat = z_flat.flatten(start_dim=1)  # [batch*pos, n_heads*d_head]
    
    # Aplatir W_O: [n_heads, d_head, d_model] -> [n_heads*d_head, d_model]
    W_O_flat = W_O.flatten(start_dim=0, end_dim=1)  # [n_heads*d_head, d_model]
    
    # F.linear fait: output = input @ weight.T + bias
    # Donc on doit transposer W_O_flat
    output_flat = F.linear(z_flat, W_O_flat.T, b_O)  # [batch*pos, d_model]
    
    # Reshape: [batch*pos, d_model] -> [batch, pos, d_model]
    output = output_flat.reshape(batch, pos, -1)
    return output


def reproduce_with_addmm(z_input, W_O, b_O, device):
    """
    Reproduit exactement le calcul de nn.Linear en utilisant torch.addmm.
    addmm fait: output = beta * input + alpha * (mat1 @ mat2)
    Pour nn.Linear: output = input @ weight.T + bias
    Donc: addmm(bias, input, weight.T, beta=1, alpha=1)
    """
    # Aplatir z: [batch, pos, n_heads, d_head] -> [batch*pos, n_heads*d_head]
    batch, pos, n_heads, d_head = z_input.shape
    z_flat = z_input.flatten(start_dim=0, end_dim=1)  # [batch*pos, n_heads, d_head]
    z_flat = z_flat.flatten(start_dim=1)  # [batch*pos, n_heads*d_head]
    
    # Aplatir W_O: [n_heads, d_head, d_model] -> [n_heads*d_head, d_model]
    W_O_flat = W_O.flatten(start_dim=0, end_dim=1)  # [n_heads*d_head, d_model]
    
    # addmm(bias, input, weight.T, beta=1, alpha=1)
    # bias doit être broadcastable: [d_model] -> [1, d_model] pour broadcasting
    bias_expanded = b_O.unsqueeze(0)  # [1, d_model]
    output_flat = torch.addmm(bias_expanded, z_flat, W_O_flat.T, beta=1.0, alpha=1.0)
    
    # Reshape: [batch*pos, d_model] -> [batch, pos, d_model]
    output = output_flat.reshape(batch, pos, -1)
    return output


def reproduce_with_matmul(z_input, W_O, b_O, device):
    """
    Reproduit le calcul en décomposant: output = (input @ weight.T) + bias
    """
    # Aplatir z: [batch, pos, n_heads, d_head] -> [batch*pos, n_heads*d_head]
    batch, pos, n_heads, d_head = z_input.shape
    z_flat = z_input.flatten(start_dim=0, end_dim=1)  # [batch*pos, n_heads, d_head]
    z_flat = z_flat.flatten(start_dim=1)  # [batch*pos, n_heads*d_head]
    
    # Aplatir W_O: [n_heads, d_head, d_model] -> [n_heads*d_head, d_model]
    W_O_flat = W_O.flatten(start_dim=0, end_dim=1)  # [n_heads*d_head, d_model]
    
    # Matmul: [batch*pos, n_heads*d_head] @ [d_model, n_heads*d_head] -> [batch*pos, d_model]
    output_flat = torch.matmul(z_flat, W_O_flat.T)  # [batch*pos, d_model]
    output_flat = output_flat + b_O  # Broadcasting
    
    # Reshape: [batch*pos, d_model] -> [batch, pos, d_model]
    output = output_flat.reshape(batch, pos, -1)
    return output

def reproduce_with_matmul_corrected(z_input, W_O, b_O, device):
    """
    Version corrigée de reproduce_with_matmul.
    Ne transpose PAS W_O_flat, ce qui correspond à l'opération einsum correcte.
    """
    # Aplatir z: [batch, pos, n_heads, d_head] -> [batch*pos, n_heads*d_head]
    batch, pos, n_heads, d_head = z_input.shape
    z_flat = z_input.flatten(start_dim=0, end_dim=1)  # [batch*pos, n_heads, d_head]
    z_flat = z_flat.flatten(start_dim=1)  # [batch*pos, n_heads*d_head]
    
    # Aplatir W_O: [n_heads, d_head, d_model] -> [n_heads*d_head, d_model]
    W_O_flat = W_O.flatten(start_dim=0, end_dim=1)  # [n_heads*d_head, d_model]
    
    # Matmul SANS transposition: [batch*pos, n_heads*d_head] @ [n_heads*d_head, d_model] -> [batch*pos, d_model]
    output_flat = torch.matmul(z_flat, W_O_flat)  # PAS de .T ici!
    output_flat = output_flat + b_O  # Broadcasting
    
    # Reshape: [batch*pos, d_model] -> [batch, pos, d_model]
    output = output_flat.reshape(batch, pos, -1)
    return output

# ============================================================================
# CODE PRINCIPAL (exécuté seulement si le script est lancé directement)
# ============================================================================

if __name__ == "__main__":
    # Configuration
    model_name = "gpt2-small"
    input_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    model_kwargs = dict(
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
        dtype=torch.float32
    )
    layer_idx = 0

    print("=" * 80)
    print("DÉBOGAGE DE nn.Linear - Reproduction exacte du calcul")
    print("=" * 80)

    # Chargement des modèles
    print("\n1. Chargement des modèles...")
    model_cpu = HookedTransformer.from_pretrained(model_name, device="cpu", **model_kwargs)
    model_mps = HookedTransformer.from_pretrained(model_name, device="mps", **model_kwargs)

    # Tokenisation
    print("2. Tokenisation...")
    tokens_cpu = model_cpu.to_tokens(input_text)
    tokens_mps = model_mps.to_tokens(input_text)

    # Exécution avec cache
    print("3. Exécution du modèle pour récupérer le cache...")
    _, cache_cpu = model_cpu.run_with_cache(tokens_cpu)
    _, cache_mps = model_mps.run_with_cache(tokens_mps)

    # Extraction des données
    hook_z_name = f"blocks.{layer_idx}.attn.hook_z"
    hook_out_name = f"blocks.{layer_idx}.hook_attn_out"

    z_cpu = cache_cpu[hook_z_name]  # [batch, pos, n_heads, d_head]
    z_mps = cache_mps[hook_z_name]

    out_cpu_real = cache_cpu[hook_out_name]
    out_mps_real = cache_mps[hook_out_name]

    # Extraction des poids
    W_O_cpu = model_cpu.blocks[layer_idx].attn.W_O  # [n_heads, d_head, d_model]
    b_O_cpu = model_cpu.blocks[layer_idx].attn.b_O  # [d_model]
    W_O_mps = model_mps.blocks[layer_idx].attn.W_O
    b_O_mps = model_mps.blocks[layer_idx].attn.b_O

    print(f"\n4. Dimensions:")
    print(f"   z shape: {z_cpu.shape}")
    print(f"   W_O shape: {W_O_cpu.shape}")
    print(f"   b_O shape: {b_O_cpu.shape}")
    print(f"   out shape: {out_cpu_real.shape}")

    # ============================================================================
    # MÉTHODE 0: Utilisation directe de nn.Linear (référence)
    # ============================================================================
    print("\n" + "=" * 80)
    print("MÉTHODE 0: Utilisation directe de nn.Linear (référence)")
    print("=" * 80)

    # Test sur CPU
    out_cpu_nn_linear = reproduce_with_nn_linear(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    diff_nn_linear_cpu = (out_cpu_nn_linear - out_cpu_real).abs()
    print(f"\nDifférence (nn.Linear CPU vs réel CPU):")
    print(f"   Max: {diff_nn_linear_cpu.max().item():.10e}")
    print(f"   Mean: {diff_nn_linear_cpu.mean().item():.10e}")

    # Test sur MPS
    out_mps_nn_linear = reproduce_with_nn_linear(z_mps, W_O_mps, b_O_mps, "mps")
    diff_nn_linear_mps = (out_mps_nn_linear - out_mps_real).abs()
    print(f"\nDifférence (nn.Linear MPS vs réel MPS):")
    print(f"   Max: {diff_nn_linear_mps.max().item():.10e}")
    print(f"   Mean: {diff_nn_linear_mps.mean().item():.10e}")

    # Comparaison CPU vs MPS
    diff_cpu_mps_nn_linear = (out_cpu_nn_linear.cpu() - out_mps_nn_linear.cpu()).abs()
    print(f"\nDifférence (nn.Linear CPU vs nn.Linear MPS):")
    print(f"   Max: {diff_cpu_mps_nn_linear.max().item():.10e}")
    print(f"   Mean: {diff_cpu_mps_nn_linear.mean().item():.10e}")

    # ============================================================================
    # MÉTHODE 1: Reproduction exacte avec torch.nn.functional.linear
    # ============================================================================
    print("\n" + "=" * 80)
    print("MÉTHODE 1: Utilisation de torch.nn.functional.linear")
    print("=" * 80)

    # Test sur CPU
    out_cpu_F_linear = reproduce_with_F_linear(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    diff_F_linear_cpu = (out_cpu_F_linear - out_cpu_real).abs()
    print(f"\nDifférence (F.linear CPU vs réel CPU):")
    print(f"   Max: {diff_F_linear_cpu.max().item():.10e}")
    print(f"   Mean: {diff_F_linear_cpu.mean().item():.10e}")

    # Test sur MPS
    out_mps_F_linear = reproduce_with_F_linear(z_mps, W_O_mps, b_O_mps, "mps")
    diff_F_linear_mps = (out_mps_F_linear - out_mps_real).abs()
    print(f"\nDifférence (F.linear MPS vs réel MPS):")
    print(f"   Max: {diff_F_linear_mps.max().item():.10e}")
    print(f"   Mean: {diff_F_linear_mps.mean().item():.10e}")

    # Comparaison CPU vs MPS
    diff_cpu_mps_F_linear = (out_cpu_F_linear.cpu() - out_mps_F_linear.cpu()).abs()
    print(f"\nDifférence (F.linear CPU vs F.linear MPS):")
    print(f"   Max: {diff_cpu_mps_F_linear.max().item():.10e}")
    print(f"   Mean: {diff_cpu_mps_F_linear.mean().item():.10e}")

    # ============================================================================
    # MÉTHODE 2: Reproduction avec torch.addmm (opération de bas niveau)
    # ============================================================================
    print("\n" + "=" * 80)
    print("MÉTHODE 2: Utilisation de torch.addmm (opération de bas niveau)")
    print("=" * 80)

    # Test sur CPU
    out_cpu_addmm = reproduce_with_addmm(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    diff_addmm_cpu = (out_cpu_addmm - out_cpu_real).abs()
    print(f"\nDifférence (addmm CPU vs réel CPU):")
    print(f"   Max: {diff_addmm_cpu.max().item():.10e}")
    print(f"   Mean: {diff_addmm_cpu.mean().item():.10e}")

    # Test sur MPS
    out_mps_addmm = reproduce_with_addmm(z_mps, W_O_mps, b_O_mps, "mps")
    diff_addmm_mps = (out_mps_addmm - out_mps_real).abs()
    print(f"\nDifférence (addmm MPS vs réel MPS):")
    print(f"   Max: {diff_addmm_mps.max().item():.10e}")
    print(f"   Mean: {diff_addmm_mps.mean().item():.10e}")

    # Comparaison CPU vs MPS
    diff_cpu_mps_addmm = (out_cpu_addmm.cpu() - out_mps_addmm.cpu()).abs()
    print(f"\nDifférence (addmm CPU vs addmm MPS):")
    print(f"   Max: {diff_cpu_mps_addmm.max().item():.10e}")
    print(f"   Mean: {diff_cpu_mps_addmm.mean().item():.10e}")

    # ============================================================================
    # MÉTHODE 3: Reproduction avec matmul + add (décomposition manuelle)
    # ============================================================================
    print("\n" + "=" * 80)
    print("MÉTHODE 3: Utilisation de matmul + add (décomposition manuelle)")
    print("=" * 80)

    # Test sur CPU
    out_cpu_matmul = reproduce_with_matmul(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    diff_matmul_cpu = (out_cpu_matmul - out_cpu_real).abs()
    print(f"\nDifférence (matmul CPU vs réel CPU):")
    print(f"   Max: {diff_matmul_cpu.max().item():.10e}")
    print(f"   Mean: {diff_matmul_cpu.mean().item():.10e}")

    # Test sur MPS
    out_mps_matmul = reproduce_with_matmul(z_mps, W_O_mps, b_O_mps, "mps")
    diff_matmul_mps = (out_mps_matmul - out_mps_real).abs()
    print(f"\nDifférence (matmul MPS vs réel MPS):")
    print(f"   Max: {diff_matmul_mps.max().item():.10e}")
    print(f"   Mean: {diff_matmul_mps.mean().item():.10e}")

    # Comparaison CPU vs MPS
    diff_cpu_mps_matmul = (out_cpu_matmul.cpu() - out_mps_matmul.cpu()).abs()
    print(f"\nDifférence (matmul CPU vs matmul MPS):")
    print(f"   Max: {diff_cpu_mps_matmul.max().item():.10e}")
    print(f"   Mean: {diff_cpu_mps_matmul.mean().item():.10e}")

    # ============================================================================
    # ANALYSE DÉTAILLÉE: Comparaison étape par étape
    # ============================================================================
    print("\n" + "=" * 80)
    print("ANALYSE DÉTAILLÉE: Comparaison étape par étape")
    print("=" * 80)

    # Comparer les inputs z
    z_diff = (z_cpu - z_mps.cpu()).abs()
    print(f"\nDifférence dans z (input):")
    print(f"   Max: {z_diff.max().item():.10e}")
    print(f"   Mean: {z_diff.mean().item():.10e}")

    # Comparer les poids W_O
    W_O_diff = (W_O_cpu - W_O_mps.cpu()).abs()
    print(f"\nDifférence dans W_O (poids):")
    print(f"   Max: {W_O_diff.max().item():.10e}")
    print(f"   Mean: {W_O_diff.mean().item():.10e}")

    # Comparer les biais b_O
    b_O_diff = (b_O_cpu - b_O_mps.cpu()).abs()
    print(f"\nDifférence dans b_O (biais):")
    print(f"   Max: {b_O_diff.max().item():.10e}")
    print(f"   Mean: {b_O_diff.mean().item():.10e}")

    # Analyse des valeurs intermédiaires pour addmm sur MPS
    print("\n" + "-" * 80)
    print("ANALYSE DES VALEURS INTERMÉDIAIRES (addmm MPS):")
    print("-" * 80)

    batch, pos, n_heads, d_head = z_mps.shape
    z_flat_mps = z_mps.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
    W_O_flat_mps = W_O_mps.flatten(start_dim=0, end_dim=1)

    print(f"z_flat_mps shape: {z_flat_mps.shape}")
    print(f"W_O_flat_mps shape: {W_O_flat_mps.shape}")
    print(f"z_flat_mps stats: min={z_flat_mps.min().item():.6f}, max={z_flat_mps.max().item():.6f}, mean={z_flat_mps.mean().item():.6f}")
    print(f"W_O_flat_mps stats: min={W_O_flat_mps.min().item():.6f}, max={W_O_flat_mps.max().item():.6f}, mean={W_O_flat_mps.mean().item():.6f}")

    # Calcul intermédiaire: matmul seulement
    matmul_intermediate_mps = torch.matmul(z_flat_mps, W_O_flat_mps.T)
    print(f"\nAprès matmul (avant add bias):")
    print(f"   Min: {matmul_intermediate_mps.min().item():.6f}")
    print(f"   Max: {matmul_intermediate_mps.max().item():.6f}")
    print(f"   Mean: {matmul_intermediate_mps.mean().item():.6f}")
    print(f"   Std: {matmul_intermediate_mps.std().item():.6f}")

    # Comparer avec CPU
    z_flat_cpu = z_cpu.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
    W_O_flat_cpu = W_O_cpu.flatten(start_dim=0, end_dim=1)
    matmul_intermediate_cpu = torch.matmul(z_flat_cpu, W_O_flat_cpu.T)

    matmul_diff = (matmul_intermediate_cpu - matmul_intermediate_mps.cpu()).abs()
    print(f"\nDifférence dans matmul intermédiaire (CPU vs MPS):")
    print(f"   Max: {matmul_diff.max().item():.10e}")
    print(f"   Mean: {matmul_diff.mean().item():.10e}")
    print(f"   Positions avec diff > 1e-5: {(matmul_diff > 1e-5).sum().item()}")
    print(f"   Positions avec diff > 1e-3: {(matmul_diff > 1e-3).sum().item()}")

    # Trouver les positions avec les plus grandes différences
    if matmul_diff.max().item() > 1e-5:
        max_diff_idx = matmul_diff.argmax()
        max_diff_pos = (max_diff_idx // matmul_diff.shape[1], max_diff_idx % matmul_diff.shape[1])
        print(f"\nPosition avec la plus grande différence:")
        print(f"   Index: {max_diff_pos}")
        print(f"   Valeur CPU: {matmul_intermediate_cpu[max_diff_pos].item():.10f}")
        print(f"   Valeur MPS: {matmul_intermediate_mps.cpu()[max_diff_pos].item():.10f}")
        print(f"   Différence: {matmul_diff[max_diff_pos].item():.10e}")

    print("\n" + "=" * 80)
    print("FIN DE L'ANALYSE")
    print("=" * 80)


# ============================================================================
# FONCTION EXPORTABLE POUR UTILISATION DANS NOTEBOOK
# ============================================================================

def debug_linear_computation(model_cpu=None, model_mps=None, cache_cpu=None, cache_mps=None, 
                             layer_idx=0, verbose=True):
    """
    Fonction exportable pour déboguer nn.Linear dans un notebook.
    
    Args:
        model_cpu: Modèle HookedTransformer sur CPU
        model_mps: Modèle HookedTransformer sur MPS
        cache_cpu: Cache du modèle CPU
        cache_mps: Cache du modèle MPS
        layer_idx: Index de la couche à déboguer
        verbose: Afficher les résultats détaillés
    
    Returns:
        dict: Dictionnaire contenant tous les résultats et différences
    """
    if model_cpu is None or model_mps is None:
        # Charger les modèles si non fournis
        model_name = "gpt2-small"
        input_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
        model_kwargs = dict(
            fold_ln=False,
            center_unembed=False,
            center_writing_weights=False,
            dtype=torch.float32
        )
        
        if model_cpu is None:
            model_cpu = HookedTransformer.from_pretrained(model_name, device="cpu", **model_kwargs)
        if model_mps is None:
            model_mps = HookedTransformer.from_pretrained(model_name, device="mps", **model_kwargs)
        
        tokens_cpu = model_cpu.to_tokens(input_text)
        tokens_mps = model_mps.to_tokens(input_text)
        
        if cache_cpu is None:
            _, cache_cpu = model_cpu.run_with_cache(tokens_cpu)
        if cache_mps is None:
            _, cache_mps = model_mps.run_with_cache(tokens_mps)
    
    # Extraction des données
    hook_z_name = f"blocks.{layer_idx}.attn.hook_z"
    hook_out_name = f"blocks.{layer_idx}.hook_attn_out"
    
    z_cpu = cache_cpu[hook_z_name]
    z_mps = cache_mps[hook_z_name]
    out_cpu_real = cache_cpu[hook_out_name]
    out_mps_real = cache_mps[hook_out_name]
    
    W_O_cpu = model_cpu.blocks[layer_idx].attn.W_O
    b_O_cpu = model_cpu.blocks[layer_idx].attn.b_O
    W_O_mps = model_mps.blocks[layer_idx].attn.W_O
    b_O_mps = model_mps.blocks[layer_idx].attn.b_O
    
    # Calculs avec différentes méthodes
    results = {}
    
    # nn.Linear
    out_cpu_nn = reproduce_with_nn_linear(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    out_mps_nn = reproduce_with_nn_linear(z_mps, W_O_mps, b_O_mps, "mps")
    results['nn_linear'] = {
        'cpu': out_cpu_nn,
        'mps': out_mps_nn,
        'diff_cpu_mps': (out_cpu_nn.cpu() - out_mps_nn.cpu()).abs()
    }
    
    # F.linear
    out_cpu_F = reproduce_with_F_linear(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    out_mps_F = reproduce_with_F_linear(z_mps, W_O_mps, b_O_mps, "mps")
    results['F_linear'] = {
        'cpu': out_cpu_F,
        'mps': out_mps_F,
        'diff_cpu_mps': (out_cpu_F.cpu() - out_mps_F.cpu()).abs()
    }
    
    # addmm
    out_cpu_addmm = reproduce_with_addmm(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    out_mps_addmm = reproduce_with_addmm(z_mps, W_O_mps, b_O_mps, "mps")
    results['addmm'] = {
        'cpu': out_cpu_addmm,
        'mps': out_mps_addmm,
        'diff_cpu_mps': (out_cpu_addmm.cpu() - out_mps_addmm.cpu()).abs()
    }
    
    # matmul
    out_cpu_matmul = reproduce_with_matmul(z_cpu, W_O_cpu, b_O_cpu, "cpu")
    out_mps_matmul = reproduce_with_matmul(z_mps, W_O_mps, b_O_mps, "mps")
    results['matmul'] = {
        'cpu': out_cpu_matmul,
        'mps': out_mps_matmul,
        'diff_cpu_mps': (out_cpu_matmul.cpu() - out_mps_matmul.cpu()).abs()
    }
    
    if verbose:
        print("Résultats du débogage:")
        for method, data in results.items():
            max_diff = data['diff_cpu_mps'].max().item()
            mean_diff = data['diff_cpu_mps'].mean().item()
            print(f"  {method:15s} - Max diff: {max_diff:.10e}, Mean diff: {mean_diff:.10e}")
    
    return results



