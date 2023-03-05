import json
import glob
import re
from loguru import logger


def extract(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = f.read()
    
    data_list = json.loads(json_data)

    output = filename.split('.')[0] + '.txt'
    with open(output, 'w', encoding='utf-8') as writer:
        for o in data_list:
            if len(o['text']) < 220:
                continue
            text = ''.join(c for c in o['text'] if c.isprintable() or c=='\n')

            # 开始处理分段，这里很多兜底策略其实在法律文本不会触发
            paras = text.split('\n')
            out_text = ''
            for p in paras:
                p = p.strip()
                if len(p) < 6:
                    continue
                # 对于下一个分段，如果不是一些序号列表，那么就真正作为新的分段了
                if not re.match('（|一|二|三|四|五|六|七|八|九|十|第|[0-9]|\(', p):
                    out_text += '\n'
                # 一句一行，这里不用正则匹配再组合因为麻烦，但是因为涉及到变量存储空间的自动扩容所以长句可能会很慢，可改多线程
                for c in p:
                    out_text += c
                    if c in ['。', '？', '！']:
                        out_text += '\n'
                        
            if out_text[-1] != '\n':
                out_text += '\n'
            writer.write(out_text)
    logger.info('Write to %s complete' % output)


def main(format):
    files = glob.glob(format)
    for f in files:
        logger.info("Start extracting data from %s." % f)
        extract(f)


if __name__=='__main__':
    main(format='*.json')
