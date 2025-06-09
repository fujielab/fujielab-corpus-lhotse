import os
import pandas as pd
from typing import List, Dict, Any 
from dataclasses import dataclass, field

class CEJCSpeakerInfo:
    """CEJC（日常会話コーパス）の話者情報を扱うクラス
    """

    def __init__(self, 
                 cejc_dir: str="/autofs/diamond2/share/corpus/CEJC2304", 
                 cejc_orig_dir: str="/autofs/diamond2/share/corpus/CEJC",
                 cejc_safia_dir: str="/autofs/diamond2/share/corpus/CEJC_safia"):
        """コンストラクタ

        Args:
            cejc_dir: CEJCのディレクトリパス（CEJC2304がある場合はそちらを指定するのが吉）
            cejc_orig_dir: CEJCのオリジナルデータのディレクトリパス（指定しない場合は cejc_dir と同じものとして扱う）
        """

        # ディレクトリパスの設定
        self.cejc_dir = cejc_dir
        
        # オリジナルデータのディレクトリパスの設定
        if cejc_orig_dir is not None:
            self.cejc_orig_dir = cejc_orig_dir
        else:
            self.cejc_orig_dir = cejc_dir

        # SAFIAデータのディレクトリパスの設定
        if cejc_safia_dir is not None:
            self.cejc_safia_dir = cejc_safia_dir
        else:
            self.cejc_safia_dir = self.cejc_orig_dir

        # メタ情報ファイルの読み込み
        self.meta_info_filepath = os.path.join(
            cejc_dir, 'metaInfo', '会話・話者・協力者等のメタ情報.xlsx'
        )
        self.df = pd.read_excel(
            self.meta_info_filepath, 
            sheet_name='話者・会話対応表')

        # 話者IDの末尾に3桁の数字がついていないものは _000 を付与        
        self.df['話者ID改'] = self.df['話者ID'].apply(
            lambda x: x + '_000' if len(x) == 4 else x
        )

        # 音声ファイル名を付与．ただし個別ICがある人のみ
        wav_filenames = []
        for i, row in self.df.iterrows():
            speaker_label = row['話者ラベル']
            session_id = row['会話ID']
            if speaker_label[:2] == "IC":
                wav_filenames.append(f"{session_id}_{speaker_label[:4]}.wav")
            else:
                wav_filenames.append(None)
        self.df['音声ファイル名'] = wav_filenames


    def get_session_id_list(self) -> List[str]:
        """会話IDのリストを取得する

        Returns:
            会話IDのリスト
        """
        return self.df['会話ID'].unique().tolist()
    
    def get_speaker_id_list(self) -> List[str]:
        """話者IDのリストを取得する

        Returns:
            話者IDのリスト
        """
        return self.df['話者ID改'].unique().tolist()
    
    def get_speaker_info_in_session(self, session_id: str) -> Dict[str, Dict[str, str]]:
        """会話IDに対する話者情報を取得する.

        Args:
            session_id: 会話ID
        
        Returns:
            results["label2id"]: 話者ラベルから話者IDへの辞書
            results["id2label"]: 話者IDから話者ラベルへの辞書
            results["id2wavfilename"]: 話者IDから音声ファイル名への辞書
        """
        subdf = self.df[self.df['会話ID'] == session_id]
        if len(subdf) == 0:
            raise ValueError('No such session id: {}'.format(session_id))
        label2id = {}
        id2label = {}
        id2wavfilename = {}
        for i, row in subdf.iterrows():
            label2id[row['話者ラベル']] = row['話者ID改']
            id2label[row['話者ID改']] = row['話者ラベル']
            id2wavfilename[row['話者ID改']] = row['音声ファイル名']
        return {
            'label2id': label2id,
            'id2label': id2label,
            'id2wavfilename': id2wavfilename
        }
    
    def get_session_data_dirpath(self, session_id: str, mode: str='none') -> str:
        """会話IDに対するデータディレクトリのパスを取得する

        Args:
            session_id: 会話ID
            mode: 'none', 'orig' または 'safia' を指定する

        Returns:
            データディレクトリのパス
        """
        if mode == 'orig':
            topdir = self.cejc_orig_dir
        elif mode == 'safia':
            topdir = self.cejc_safia_dir
        else:
            topdir = self.cejc_dir
        subject_id = session_id[:4]
        session_base_id = session_id[:8]
        return os.path.join(
            topdir, 'data', subject_id, session_base_id)

    def get_session_suw_filepath(self, session_id: str) -> str:
        """会話IDに対するSUWファイルのパスを取得する

        Args:
            session_id: 会話ID

        Returns:
            SUWファイルのパス
        """
        dirpath = self.get_session_data_dirpath(session_id)
        return os.path.join(dirpath, session_id + '-morphSUW.csv')

    def get_session_luw_filepath(self, session_id: str) -> str:
        """会話IDに対するLUWファイルのパスを取得する

        Args:
            session_id: 会話ID

        Returns:
            LUWファイルのパス
        """
        dirpath = self.get_session_data_dirpath(session_id)
        return os.path.join(dirpath, session_id + '-morphLUW.csv')

    def get_session_wav_filepath(self, session_id: str, speaker_id: str, mode: str='orig') -> str:
        """会話IDに対する音声ファイルのパスを取得する

        Args:
            session_id: 会話ID
            speaker_id: 話者ID
            mode: 'none', 'orig' または 'safia' を指定する
        
        Returns:
            音声ファイルのパス
        """
        row = self.df[(self.df['会話ID'] == session_id) & (self.df['話者ID改'] == speaker_id)]
        if len(row) == 0:
            raise ValueError('No such condition (session_id: {}, speaker_id: {})'.format(session_id, speaker_id))
        wav_filename = row['音声ファイル名'].values[0]
        if wav_filename is not None:
            return os.path.join(
                self.get_session_data_dirpath(session_id, mode),
                wav_filename)
        else:
            return None
        
        """
        speaker_info = self.get_speaker_info_in_session(session_id)
        wav_filename = speaker_info['id2wavfilename'][speaker_id]
        if wav_filename is not None:
            return os.path.join(
                self.get_session_data_dirpath(session_id, orig_flag=True),
                wav_filename)
        else:
            return None
        """

@dataclass
class MorphInfo:
    """形態素情報"""
    text: str # 書字形
    pron: str # 発音形
    is_filler: bool = False     # フィラーかどうか
    is_disfluency: bool = False # 非流暢（言いよどみ）かどうか
    has_pause: bool = False     # 次に休止があるか
    has_border: bool = False    # 次に文境境界があるか
    is_privacy: bool = False    # 仮名かどうか


@dataclass
class UtteranceInfo:
    speaker_label: str           # 話者ラベル（IC01_玲子 など）
    speaker_id: str              # 話者ID（C001_000 など）
    start_time: float            # 発話開始時刻
    end_time: float              # 発話終了時刻
    utterance_id: str = ''       # ESPnetで使うための発話ID
    morphs: List[MorphInfo] = field(default_factory=list) # 形態素情報のリスト
    utterance_text: str = ''     # 発話テキスト（文節区切りなし）
    utterance_pron: str = ''     # 発話発音（モーラ単位，文節区切り，特殊記号あり）

class CEJCTextData:
    """CEJC（日常会話コーパス）のテキストデータを扱うクラス"""
    def __init__(self, 
                 suw_filename: str, 
                 luw_filename: str,
                 speaker_info: Dict[str, Dict[str, str]],
                 min_gap_dur: float=0.3, 
                 max_utt_dur: float=10.0):
        """コンストラクタ

        Args:
            filename: SUWファイルのパス
            min_gap_dur: 発話間の最小間隔時間（秒）
            max_utt_dur: 発話の最大時間（秒）
        """
        self.suw_fileanme = suw_filename
        self.luw_filename = luw_filename
        self.speaker_info = speaker_info
        self.min_gap_dur = min_gap_dur
        self.max_utt_dur = max_utt_dur

        self.text_info = self._construct_text_info()

    def _construct_text_info(self):
        """テキスト情報を構築する"""

        label2id = self.speaker_info['label2id']
        id2wavfilename = self.speaker_info['id2wavfilename']
        suw_df = pd.read_csv(self.suw_fileanme, encoding='shift-jis')
        luw_df = pd.read_csv(self.luw_filename, encoding='shift-jis')

        # 話者ラベルのユニークなリストを取得
        speaker_labels = suw_df['話者ラベル'].unique()

        # Step 1. SUW情報を元に基本的な発話情報を構築
        #         また，LUW情報を参照しながら文節区切りをつける
        utterance_info_list = []
        # 話者ラベルごとに処理
        for speaker_label in speaker_labels:
            subdf_suw = suw_df[suw_df['話者ラベル'] == speaker_label]
            subdf_luw = luw_df[luw_df['話者ラベル'] == speaker_label]

            speaker_id = label2id[speaker_label]
            wav_filename = id2wavfilename[speaker_id]
            if wav_filename is None:
                # 警告を表示
                print('Warning: No wav filename for speaker {} ({})'.format(speaker_id, speaker_label), flush=True)
                continue

            current_utterance_info = None
            luw_index = -1
            luw_pron = ''
            for suw_index, row in subdf_suw.iterrows():
                speaker_label = row['話者ラベル']
                start_time = row['転記単位の開始時刻']
                end_time = row['転記単位の終了時刻']
                pron = row['発音']

                if pron.startswith('('):
                    pron = pron[1:-1]
                    if current_utterance_info is not None and \
                        len(current_utterance_info.morphs) > 0 and \
                        current_utterance_info.morphs[-1].pron == pron:
                            continue
                
                morph_info = MorphInfo(
                    text=row['書字形'], 
                    pron=pron,
                    is_filler=row['品詞'] == '感動詞-フィラー', 
                    is_disfluency=row['品詞'] == '言いよどみ',
                    is_privacy=row['仮名'] == 1)
                
                if luw_pron == '':
                    luw_index += 1
                    if luw_index < len(subdf_luw):
                        luw_pron = subdf_luw['発音'].iloc[luw_index]
                        has_border = subdf_luw['文節頭フラグ'].iloc[luw_index] == 'B'
                        if '(' in luw_pron:
                            next_luw_pron = subdf_luw['発音'].iloc[luw_index + 1]
                            while '(' in luw_pron and '(' in next_luw_pron and \
                                (luw_pron in next_luw_pron or next_luw_pron in luw_pron):
                                luw_pron = next_luw_pron if len(next_luw_pron) > len(luw_pron) else luw_pron
                                luw_index += 1
                                next_luw_pron = subdf_luw['発音'].iloc[luw_index + 1]
                        if '(' in luw_pron:
                            luw_pron = luw_pron.replace('(', '').replace(')', '')
                        
                        # while luw_pron.startswith('(') and subdf_luw['発音'].iloc[luw_index + 1].startswith(luw_pron):
                        #     luw_index += 1
                        #     luw_pron = subdf_luw['発音'].iloc[luw_index]
                        # if luw_pron.startswith('('):
                        #     luw_pron = luw_pron.replace('(', '').replace(')', '')
                        
                        if has_border and \
                           current_utterance_info is not None and \
                                len(current_utterance_info.morphs) > 0:
                                current_utterance_info.morphs[-1].has_border = True
                    else:
                        luw_pron = ''
                
                if luw_pron.startswith(morph_info.pron):
                    luw_pron = luw_pron[len(morph_info.pron):]
                elif pron == '◇':
                    continue
                else:
                    # 警告を表示
                    print('Warning: LUW and SUW are not matched: {} vs. {} at {}'.format(luw_pron, morph_info.pron, (speaker_label, start_time, end_time)), flush=True)
                    pass
                
                if current_utterance_info is None:
                    current_utterance_info = UtteranceInfo(
                        speaker_label=speaker_label,
                        speaker_id=label2id[speaker_label],
                        start_time=start_time,
                        end_time=end_time,
                        morphs=[morph_info])
                else:
                    is_in_sentence = row['文頭フラグ'] == 'I'
                    gap = start_time - current_utterance_info.end_time
                    is_speaker_changed = speaker_label != current_utterance_info.speaker_label  # this is always False
                    is_in_same_trans_unit = end_time == current_utterance_info.end_time
                    is_under_min_gap = start_time - current_utterance_info.end_time < self.min_gap_dur
                    is_over_max_utt = end_time - current_utterance_info.start_time > self.max_utt_dur

                    # 接続する条件:
                    #    文内なら接続
                    #    話者が変わってない場合，ギャップ未満の無音時間で，接続した際に最大発話時間を超えない場合
                    # if  is_in_sentence or \
                    if is_in_same_trans_unit or \
                    (not is_speaker_changed and \
                        is_under_min_gap and not is_over_max_utt):
                        # ポーズがある場合はフラグを立てる
                        if start_time - current_utterance_info.end_time > 0.1:
                            morph_info.has_pause = True
                        # 終了時間を更新
                        current_utterance_info.end_time = end_time
                        # 形態素情報を追加
                        current_utterance_info.morphs.append(morph_info)                 
                    else:
                        # 接続しない
                        utterance_info_list.append(current_utterance_info)
                        current_utterance_info = UtteranceInfo(
                            speaker_label=speaker_label,
                            speaker_id=label2id[speaker_label],
                            start_time=start_time,
                            end_time=end_time,
                            morphs=[morph_info])
            if current_utterance_info is not None:
                utterance_info_list.append(current_utterance_info)
        
        # # Step 2. 形態素情報の修正をする
        # # 発音に '(ロギーン)' のように括弧がついている場合，
        # # 同じ内容を持つ後続の形態素情報は削除し，()記号も削除する
        # for suw_index in range(len(utterance_info_list)):
        #     morphs = utterance_info_list[suw_index].morphs
        #     new_morphs = []
        #     current_morph = None
        #     for j in range(len(morphs)):
        #         morph = morphs[j]
        #         if morph.pron.startswith('(') and morph.pron.endswith(')'):
        #             if current_morph is None:
        #                 current_morph = morph
        #                 current_morph.pron = current_morph.pron[1:-1]
        #             else:
        #                 current_morph.text += morph.text
        #         else:
        #             if current_morph is not None:
        #                 new_morphs.append(current_morph)
        #                 current_morph = None
        #             new_morphs.append(morph)
        #     if current_morph is not None:
        #         new_morphs.append(current_morph)
        #     utterance_info_list[suw_index].morphs = new_morphs

        # # Step 3. LUW情報を元に文節区切りをつける
        # # luw_df = pd.read_csv(self.luw_filename, encoding='shift-jis')
        # for suw_index in range(len(utterance_info_list)):
        #     speaker_label = utterance_info_list[suw_index].speaker_label
        #     start_time = utterance_info_list[suw_index].start_time
        #     end_time = utterance_info_list[suw_index].end_time
        #     morphs = utterance_info_list[suw_index].morphs

        #     # print("###")
        #     # print(speaker_label, start_time, end_time)

        #     # 当該話者の当該発話単位のLUW情報を取得
        #     subdf_suw = luw_df[(luw_df['話者ラベル'] == speaker_label) & \
        #                    (luw_df['転記単位の開始時刻'] >= start_time) & \
        #                    (luw_df['転記単位の終了時刻'] <= end_time)]
        #     # # subdfの最後の終了時刻が end_time と等しく無い場合は，もう一つ後ろのLUW情報を取得
        #     # # たぶん．1つで十分だと思うが例外もあるかも...
        #     # if len(subdf_suw) > 0 and subdf_suw['転記単位の終了時刻'].iloc[-1] != end_time:
        #     #     last_index = subdf_suw.index[-1]
        #     #     next_subdf = luw_df.iloc[last_index + 1:last_index + 2]
        #     #     subdf_suw = pd.concat([subdf_suw, next_subdf], axis=0)

        #     if len(subdf_suw) == 0:
        #         # 警告を表示
        #         print('Warning: No LUW information for speaker {} at {}-{}'.format(speaker_label, start_time, end_time), flush=True)
        #         continue

        #     luw_index = 0
        #     # print(subdf)
        #     luw_pron = subdf_suw['発音'].iloc[0]
        #     luw_brace_flag = False
        #     if luw_pron.startswith('('):
        #         luw_pron = luw_pron[1:-1]
        #         luw_brace_flag = True
        #     for suw_index in range(len(morphs)):
        #         suw_pron = morphs[suw_index].pron
        #         for suw_pron_char in suw_pron:
        #             if luw_pron[0] == suw_pron_char:
        #                 luw_pron = luw_pron[1:]
        #             else:
        #                 # SUW側に余計な文字が入っていることがあるのでスキップ
        #                 # 警告を表示
        #                 print('Warning: LUW and SUW are not matched: {} vs. {} at {}'.format(luw_pron, suw_pron, (speaker_label, start_time, end_time)), flush=True)
        #                 # print(morphs)
        #                 # print(speaker_label, suw_index, suw_pron)
        #                 # print(luw_pron)
        #                 # print(subdf)
        #                 continue
        #         if len(luw_pron) == 0:
        #             if luw_brace_flag:
        #                 luw_index += 1
        #                 while luw_index < len(subdf_suw) - 1 and subdf_suw['発音'].iloc[luw_index].startswith('('):
        #                     luw_index += 1
        #                 luw_brace_flag = False
        #             else:
        #                 luw_index += 1
                    
        #             if luw_index < len(subdf_suw):
        #                 luw_pron = subdf_suw['発音'].iloc[luw_index]
        #                 if luw_pron.startswith('('):
        #                     luw_pron = luw_pron[1:-1]
        #                     luw_brace_flag = True
        #                 if subdf_suw['文節頭フラグ'].iloc[luw_index] == 'B':
        #                     morphs[suw_index].has_border = True
        #             else:
        #                 luw_pron = ''
        
        # Step. 4. 発話テキストを生成
        for suw_index in range(len(utterance_info_list)):
            morphs = utterance_info_list[suw_index].morphs
            tokens = []
            for j in range(len(morphs)):
                if morphs[j].is_disfluency:
                    continue
                tokens_ = [morphs[j].text]
                if morphs[j].is_privacy:
                    tokens_ = ['<mask>']
                if morphs[j].is_filler:
                    tokens_ = ['<F>'] + tokens_ + ['</F>']
                if morphs[j].has_pause:
                    # tokens_.append('、') 
                    tokens_.insert(0, '、')
                tokens_ = [t.replace('◇', '') for t in tokens_]
                tokens_ = [t for t in tokens_ if t != '']
                if '<mask>' in tokens_ and len(tokens) > 0 and tokens[-1] == '<mask>':
                    continue
                if len(tokens_) > 0:
                    tokens.extend(tokens_)
            utterance_info_list[suw_index].utterance_text = ' '.join(tokens)
        
        # Step 5. 発話発音を生成
        for suw_index in range(len(utterance_info_list)):
            morphs = utterance_info_list[suw_index].morphs
            tokens = []            
            for j in range(len(morphs)):
                moras = split_pront_to_mora(morphs[j].pron)

                if morphs[j].is_disfluency:
                    moras = [x + '+D' for x in moras]
                elif morphs[j].is_filler:
                    moras = [x + '+F' for x in moras]

                if morphs[j].is_privacy:
                    moras = ['<mask>']

                if morphs[j].has_pause:
                    # moras.append('<sp>')
                    moras.insert(0, '<sp>')
                elif morphs[j].has_border and len(moras) > 0:
                    moras += ['|']

                tokens.extend(moras)
            
            while len(tokens) > 0 and tokens[-1] == '|':
                tokens.pop()

            utterance_info_list[suw_index].utterance_pron = ' '.join(tokens) 

        # Step 6. 発話IDを付与
        label2id = self.speaker_info['label2id']
        id2wavfilename = self.speaker_info['id2wavfilename']
        for suw_index in range(len(utterance_info_list)):
            speaker_id = label2id[utterance_info_list[suw_index].speaker_label]
            wav_filename = id2wavfilename[speaker_id]
            if wav_filename is None:
                # 警告を表示
                # print('Warning: No wav filename for speaker {}'.format(speaker_id))
                continue
            wav_filename = wav_filename.replace('.wav', '')
            start_time = int(utterance_info_list[suw_index].start_time * 1000)
            end_time = int(utterance_info_list[suw_index].end_time * 1000)

            utterance_info_list[suw_index].utterance_id = \
                f"{speaker_id}_{wav_filename}_{start_time:07d}_{end_time:07d}"
            
        # Step 7. 0.1秒未満，10秒以上，発音形が空のものを削除
        utterance_info_list_ = []
        for utterance_info in utterance_info_list:
            duration = utterance_info.end_time - utterance_info.start_time
            if duration < 0.1:
                # 警告を表示
                print('Warning: Too SHORT utterance: {}: {:7.3f}s {}'.format(utterance_info.utterance_id, duration, utterance_info.utterance_pron), flush=True)
                continue

            if duration > 10.0:
                # 警告を表示
                print('Warning: Too LONG  utterance: {}: {:7.3f}s {}'.format(utterance_info.utterance_id, duration, utterance_info.utterance_pron), flush=True)
                continue

            if len(utterance_info.utterance_pron) < 1: 
                # 警告を表示
                print('Warning: No pron: {}, {:7.3f}s'.format(utterance_info.utterance_id, duration), flush=True)
                continue

            utterance_info_list_.append(utterance_info)


        utterance_info_list = utterance_info_list_

        return utterance_info_list

##
_kana_list = None
_kanas = """ア イ ウ エ オ カ キ ク ケ コ ガ ギ グ ゲ ゴ サ シ ス セ ソ
ザ ジ ズ ゼ ゾ タ チ ツ テ ト ダ デ ド ナ ニ ヌ ネ ノ ハ ヒ フ ヘ ホ
バ ビ ブ ベ ボ パ ピ プ ペ ポ マ ミ ム メ モ ラ リ ル レ ロ ヤ ユ ヨ
ワ ヲ ン ウィ ウェ ウォ キャ キュ キョ ギャ ギュ ギョ シャ シュ ショ
ジャ ジュ ジョ チャ チュ チョ ディ ドゥ デュ ニャ ニュ ニョ ヒャ ヒュ ヒョ
ビャ ビュ ビョ ピャ ピュ ピョ ミャ ミュ ミョ リャ リュ リョ イェ クヮ
グヮ シェ ジェ ティ トゥ チェ ツァ ツィ ツェ ツォ ヒェ ファ フィ フェ フォ フュ
テュ ブィ ニェ ミェ スィ ズィ ヴァ ヴィ ヴ ヴェ ヴォ ー ッ | <sp>"""
_kana_list = [x.replace(' ', '') for x in _kanas.replace('\n', ' ').split(' ')]
_kana_list = sorted(_kana_list, key=len, reverse=True)

def split_pront_to_mora(pron: str) -> List[str]:
    """発音形をモーラに分割する

    Args:
        pron: 発音形
    
    Returns:
        モーラのリスト
    """
    moras = []
    while len(pron) > 0:
        flag = False
        for kana in _kana_list:
            if pron.startswith(kana):
                moras.append(kana)
                pron = pron[len(kana):]
                flag = True
                break
        if not flag:
            # 警告を表示する
            # print('Warning: Unknown kana: {}'.format(pron))
            # pronの先頭文字を削除する
            pron = pron[1:]
    return moras

#
def run_safia(cejc_dir: str, cejc_safia_dir: str, overwrite: bool=False):
    cejc_speaker_info = CEJCSpeakerInfo(cejc_dir)
    session_id_list = cejc_speaker_info.get_session_id_list()

    import tqdm
    import wave
    import numpy as np
    from .safia import apply_safia

    with tqdm.tqdm(session_id_list) as pbar:
        for session_id in session_id_list:
            pbar.set_postfix({"session_id": session_id}, refresh=True)
            id2wavdata = {}
            wav_len = -1
            speaker_info = cejc_speaker_info.get_speaker_info_in_session(session_id)

            wavfilename_list = list(speaker_info['id2wavfilename'].values())
            wavfilename_list = [x for x in wavfilename_list if x is not None]

            for speaker_id in speaker_info['id2wavfilename']:
                print(f"session_id: {session_id}, speaker_id: {speaker_id}", flush=True)
                wav_path = cejc_speaker_info.get_session_wav_filepath(session_id, speaker_id, mode='orig')
                if wav_path is None:
                    print(f"Warning: No wav file for {session_id}, {speaker_id}", flush=True)
                    continue

                try:
                    import librosa
                    y, sr = librosa.load(wav_path, sr=16000, mono=True)
                    data = (y * 32767).astype(np.int16).tobytes()
                except Exception as e:
                    print(f"Error loading {wav_path}: {e}", flush=True)
                    continue
                
                x = np.frombuffer(data, dtype=np.int16)
                if wav_len == -1:
                    wav_len = len(x)
                elif wav_len > len(x):
                    print(f"Warning: {session_id}, {speaker_id} has shorter wav length than others: {len(x)} < {wav_len}", flush=True)
                    x = np.pad(x, (0, wav_len - len(x)), mode='constant')
                elif wav_len < len(x):
                    print(f"Warning: {session_id}, {speaker_id} has longer wav length than others: {len(x)} > {wav_len}", flush=True)
                    x = x[:wav_len]

                id2wavdata[speaker_id] = x

            if len(id2wavdata) == 0:
                print(f"Warning: No valid wav data for {session_id}", flush=True)
                pbar.update(1)
                continue
            elif len(id2wavdata) > 1:
                x = np.stack(list(id2wavdata.values()), axis=0)
                x_safia = apply_safia(x)
            else:
                x_safia = x.reshape(1, -1)

            for i, speaker_id in enumerate(id2wavdata):
                wav_path = cejc_speaker_info.get_session_wav_filepath(session_id, speaker_id)
                rel_wav_path = wav_path[len(cejc_dir):]
                if rel_wav_path.startswith('/'):
                    rel_wav_path = rel_wav_path[1:]
                out_wav_path = os.path.join(cejc_safia_dir, rel_wav_path)
                
                os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
                if not overwrite and os.path.exists(out_wav_path):
                    print(f"Skip: {out_wav_path} already exists", flush=True)
                    continue
                with wave.open(out_wav_path, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(x_safia[i].astype(np.int16).tobytes())

            pbar.update(1)


if __name__ == "__main__":
    # CEJCの話者情報を取得
    cejc = CEJCSpeakerInfo()
    print(cejc.get_session_id_list())
    print(cejc.get_speaker_id_list())
    
    # 特定の会話IDに対する話者情報を取得
    session_id = cejc.get_session_id_list()[0]
    speaker_info = cejc.get_speaker_info_in_session(session_id)
    print(speaker_info)

    # SUWファイルとLUWファイルのパスを取得
    suw_filepath = cejc.get_session_suw_filepath(session_id)
    luw_filepath = cejc.get_session_luw_filepath(session_id)
    print(suw_filepath, luw_filepath)

    # 話者IDに対する音声ファイルのパスを取得
    speaker_id = 'C001_000'
    wav_filepath = cejc.get_session_wav_filepath(session_id, speaker_id, mode='orig')
    print(wav_filepath)
