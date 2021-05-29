import json

from modules.samples import all_labels, FILE_PATTERNS, SAMPLES_PATH

positives = {'positives': ['/content/drive/MyDrive/alina_clean-2/samples/0/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/pos_clean_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/pos_clean_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/pos_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/pos_noisy_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/pos_noisy_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/1/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/1/pos_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/pos_clean_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/pos_clean_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/pos_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/pos_noisy_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/pos_noisy_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/3/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/3/pos_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/4/invalid_pos_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/4/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/5/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/5/pos_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/6/pos_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/6/pos_noisy_0.wav', ]}
negatives = {'negatives': ['/content/drive/MyDrive/alina_clean-2/samples/0/neg_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_clean_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_clean_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_noisy_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_noisy_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_random_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_random_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/0/neg_random_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/1/neg_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/1/neg_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/1/neg_random_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_clean_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_clean_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_noisy_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_noisy_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_random_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_random_1.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/2/neg_random_2.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/3/neg_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/3/neg_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/3/neg_random_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/4/invalid_neg_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/4/neg_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/4/neg_random_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/5/neg_clean_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/5/neg_noisy_0.wav',
                           '/content/drive/MyDrive/alina_clean-2/samples/5/neg_random_0.wav', ]}


def save_labels(file_path=None):
    for key, items in all_labels.items():
        print('GROUP:', key)
        for item in items:
            print(item['path'])
        print()

    print('TOTAL FILES IN GROUPS:')
    for key in FILE_PATTERNS:
        print(key, '->', len(all_labels[key]))

    file_path = file_path or (SAMPLES_PATH / 'meta.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(all_labels, file, ensure_ascii=False)

    print('\nALL LABELS SAVED TO', file_path)
