import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ColabArchiveAcceptanceTests(unittest.TestCase):
    def test_packager_offers_separate_source_and_data_archives(self):
        from tools import notebook06_bundle
        self.assertTrue(callable(getattr(notebook06_bundle, 'build_artifacts', None)),
                        '소스와 데이터 별도 아카이브 생성 기능이 필요합니다')

    def test_notebook_setup_no_longer_embeds_large_data(self):
        notebook = json.loads((ROOT / 'ipynb/06_expert80_benchmark.ipynb').read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare_setup') == 'setup')
        source = ''.join(cell['source'])
        self.assertLess(len(source), 100_000, '설정 셀에 대용량 자료가 남아 있습니다')
        self.assertNotIn('_BUNDLE_DATA', source)


if __name__ == '__main__':
    unittest.main()
