import json
from typing import List, Dict, Any

class CodeNetDataProcessor:
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats = {
            'total_pairs': len(data),
            'clone_pairs': 0,
            'nonclone_pairs': 0,
            'language_pairs': {},
            'languages': set()
        }
        
        for item in data:
            if item['type'] == 'clone':
                stats['clone_pairs'] += 1
            else:
                stats['nonclone_pairs'] += 1
            
            lang1, lang2 = item['ll1'], item['ll2']
            stats['languages'].add(lang1)
            stats['languages'].add(lang2)
            
            lang_pair = f"{min(lang1, lang2)}_{max(lang1, lang2)}"
            if lang_pair not in stats['language_pairs']:
                stats['language_pairs'][lang_pair] = {'clone': 0, 'nonclone': 0}
            stats['language_pairs'][lang_pair][item['type']] += 1
        
        stats['languages'] = list(stats['languages'])
        
        return stats