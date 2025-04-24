import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import torch
from torch import nn

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SaveManager:
    """Gerencia o salvamento e carregamento de modelos IA com segurança"""
    
    def __init__(self, save_dir: str = "saved_models") -> None:
        self.save_dir = save_dir
        self.best_model_dir = os.path.join(save_dir, "best_models")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        self._setup_safe_loading()

    def _setup_safe_loading(self) -> None:
        """Configura globais seguros para torch.load"""
        try:
            from numpy.core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
        except ImportError as e:
            logger.warning(f"Não foi possível configurar carregamento seguro: {e}")

    def _get_timestamp(self) -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_model(self, model: nn.Module, model_name: str = "model", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Salva o modelo com metadados
        
        Args:
            model: Modelo PyTorch a ser salvo
            model_name: Nome base para o arquivo
            metadata: Dicionário com metadados adicionais
            
        Returns:
            Caminho para o arquivo do modelo salvo
        """
        timestamp = self._get_timestamp()
        model_path = os.path.join(self.save_dir, f"{model_name}_{timestamp}.pt")
        metadata_path = os.path.join(self.save_dir, f"{model_name}_{timestamp}.json")

        # Garante metadados mínimos
        default_metadata = {
            'save_date': timestamp,
            'model_name': model_name,
            'framework': 'pytorch'
        }
        if metadata is not None:
            if not isinstance(metadata, dict):
                logger.warning("Metadados fornecidos não são um dicionário. Serão ignorados.")
                metadata = {}
            default_metadata.update(metadata)

        # Salva de forma segura
        torch.save(model.state_dict(), model_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(default_metadata, f, indent=4)

        return model_path

    def load_model(self, model: nn.Module, model_path: str) -> bool:
        """Carrega um modelo específico com verificação de segurança
        
        Args:
            model: Modelo PyTorch onde os pesos serão carregados
            model_path: Caminho para o arquivo do modelo
            
        Returns:
            True se o carregamento foi bem-sucedido, False caso contrário
        """
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))
            return True
        except Exception as e:
            logger.error(f"Falha ao carregar modelo: {e}")
            return False

    def save_best_model(self, model: nn.Module, score: float, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Salva o modelo apenas se for melhor que os existentes
        
        Args:
            model: Modelo PyTorch a ser salvo
            score: Pontuação do modelo
            metadata: Metadados adicionais
            
        Returns:
            Caminho para o arquivo do modelo salvo
        """
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            logger.warning("Metadados fornecidos não são um dicionário. Serão ignorados.")
            metadata = {}
        
        metadata.update({
            'score': float(score),
            'save_date': self._get_timestamp(),
            'is_best': True
        })

        filename = f"best_model_score_{score:.2f}_{self._get_timestamp()}.pt"
        model_path = os.path.join(self.best_model_dir, filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, model_path)

        self._cleanup_old_models()
        return model_path

    def load_best_model(self, model: nn.Module) -> Tuple[bool, Dict[str, Any]]:
        """Carrega o melhor modelo disponível
        
        Args:
            model: Modelo PyTorch onde os pesos serão carregados
            
        Returns:
            Tupla contendo:
            - bool: True se o carregamento foi bem-sucedido
            - dict: Metadados do modelo carregado (vazio se falhou)
        """
        try:
            model_files = [f for f in os.listdir(self.best_model_dir) 
                        if f.startswith('best_model_') and f.endswith('.pt')]
            
            if not model_files:
                return False, {}  # Retorna dicionário vazio

            best_model = None
            best_score = -float('inf')
            best_metadata = {}

            for model_file in model_files:
                try:
                    model_path = os.path.join(self.best_model_dir, model_file)
                    data = torch.load(model_path, weights_only=True, map_location='cpu')
                    
                    if not isinstance(data, dict):
                        continue
                        
                    file_metadata = data.get('metadata', {})
                    if not isinstance(file_metadata, dict):
                        file_metadata = {}
                    
                    file_score = float(file_metadata.get('score', -float('inf')))
                    
                    if file_score > best_score:
                        best_score = file_score
                        best_model = model_path
                        best_metadata = file_metadata
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar {model_file}: {str(e)}")
                    continue

            if best_model:
                data = torch.load(best_model, weights_only=True, map_location='cpu')
                model.load_state_dict(data['model_state_dict'])
                return True, best_metadata

            return False, {}
            
        except Exception as e:
            logger.error(f"Erro crítico ao carregar modelo: {str(e)}")
            return False, {}

    def _cleanup_old_models(self, keep: int = 3) -> None:
        """Mantém apenas os 'keep' melhores modelos
        
        Args:
            keep: Número de modelos a manter
        """
        model_files = [f for f in os.listdir(self.best_model_dir) 
                    if f.startswith('best_model_') and f.endswith('.pt')]
        
        if len(model_files) <= keep:
            return

        # Extrai scores dos nomes dos arquivos
        models_with_scores = []
        for f in model_files:
            try:
                score = float(f.split('_score_')[1].split('_')[0])
                models_with_scores.append((f, score))
            except (IndexError, ValueError):
                continue

        # Ordena do melhor para o pior
        models_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Remove os mais antigos além do limite
        for f, _ in models_with_scores[keep:]:
            try:
                os.remove(os.path.join(self.best_model_dir, f))
            except OSError as e:
                logger.error(f"Erro ao remover modelo antigo {f}: {e}")
                
    def cleanup_unused_models(self, current_agents: List[int], model_prefix: str = "evolutionary_trash_collector_agent") -> None:
        """Remove modelos de agentes não utilizados
        
        Args:
            current_agents: Lista de IDs de agentes ativos
            model_prefix: Prefixo usado nos nomes dos arquivos de modelo
        """
        try:
            # Lista todos os arquivos de modelo
            model_files = [f for f in os.listdir(self.save_dir) 
                        if f.startswith(model_prefix) and f.endswith('.pt')]
            
            # Identifica os agentes atuais (arquivos que devem ser mantidos)
            current_files = [f"{model_prefix}_{i}_" for i in current_agents]
            
            for model_file in model_files:
                # Verifica se o arquivo não pertence a um agente atual
                if not any(prefix in model_file for prefix in current_files):
                    try:
                        os.remove(os.path.join(self.save_dir, model_file))
                        # Remove também o arquivo de metadados correspondente
                        metadata_file = model_file.replace('.pt', '.json')
                        if os.path.exists(os.path.join(self.save_dir, metadata_file)):
                            os.remove(os.path.join(self.save_dir, metadata_file))
                        logger.info(f"Removido modelo não utilizado: {model_file}")
                    except OSError as e:
                        logger.error(f"Erro ao remover {model_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Erro na limpeza de modelos: {e}")


class CheckpointManager:
    """Gerencia pontos de recuperação do treinamento"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_saves: int = 5) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.max_saves = max_saves
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._setup_safe_loading()

    def _setup_safe_loading(self) -> None:
        """Configura globais seguros para torch.load"""
        try:
            from numpy.core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
        except ImportError as e:
            logger.warning(f"Não foi possível configurar carregamento seguro: {e}")

    def save_checkpoint(self, trainer: Any, episode: Optional[int] = None) -> str:
        """Salva estado completo do treinamento
        
        Args:
            trainer: Objeto trainer contendo o modelo e otimizador
            episode: Número do episódio atual (opcional)
            
        Returns:
            Caminho para o arquivo de checkpoint salvo
        """
        checkpoint = {
            'model_state': trainer.agent.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'epsilon': trainer.epsilon,
            'episode': episode if episode is not None else trainer.steps_done,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'scores': getattr(trainer, 'current_scores', []),
            'metadata': {
                'framework': 'pytorch',
                'training_phase': 'intermediate'
            }
        }

        filename = f"checkpoint_ep{checkpoint['episode']}_{checkpoint['timestamp']}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save(checkpoint, path)
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint salvo em {path}")
        return path

    def load_latest_checkpoint(self, trainer: Any) -> bool:
        """Carrega o checkpoint mais recente
        
        Args:
            trainer: Objeto trainer onde o estado será carregado
            
        Returns:
            True se o carregamento foi bem-sucedido, False caso contrário
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                    if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoints:
            return False

        # Encontra o checkpoint mais recente
        latest = max(checkpoints, key=lambda x: os.path.getmtime(
            os.path.join(self.checkpoint_dir, x)))
        
        path = os.path.join(self.checkpoint_dir, latest)
        
        try:
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            
            trainer.agent.load_state_dict(checkpoint['model_state'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            trainer.epsilon = checkpoint['epsilon']
            trainer.steps_done = checkpoint.get('episode', trainer.steps_done)
            
            if hasattr(trainer, 'current_scores'):
                trainer.current_scores = checkpoint.get('scores', [])
            
            logger.info(f"Checkpoint carregado: {latest}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}")
            return False

    def _cleanup_old_checkpoints(self) -> None:
        """Remove checkpoints antigos mantendo apenas os mais recentes"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                    if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if len(checkpoints) <= self.max_saves:
            return

        # Ordena por data de modificação (mais antigo primeiro)
        checkpoints.sort(key=lambda x: os.path.getmtime(
            os.path.join(self.checkpoint_dir, x)))
        
        # Remove os mais antigos
        for old_checkpoint in checkpoints[:-self.max_saves]:
            try:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
            except OSError as e:
                logger.error(f"Erro ao remover checkpoint antigo {old_checkpoint}: {e}")