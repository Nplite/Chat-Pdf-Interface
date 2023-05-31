!pip install ttsmms
!curl https://dl.fbaipublicfiles.com/mms/tts/eng.tar.gz --output eng.tar.gz
!mkdir -p data && tar -xzf eng.tar.gz -C data/
from ttsmms import TTS

tts = TTS("data/eng")
wav = tts.synthesis("रोजच्या वापरातले इंग्रजी वाक्य मराठी अर्थासह")

from IPython.display import Audio
Audio(wav["x"], rate=wav["sampling_rate"])

text = """कोणत्याही भाषेवर प्रभुत्व मिळवायच असेल तर आपल्याकडे विपुल शब्दसंग्रह असायला हवा.  इंग्रजी चा मोठया प्रमाणावर वापर पाहता आपल्या ला काही कॉमन शब्दांचा ,दैनंदिन जीवनात वापर होणाऱ्या शब्दांचा , वाक्यांचा अर्थ तर नक्की माहीत असायला हवा.

अक्षरपासून शब्द बनतात आणि शब्दांपासून वाक्य , त्यामुळे इंग्रजीत असणारी रोजच्या  वापरातली काही शब्द माहिती असणे नक्की उपयोगी पडेल"""

wav_text =tts.synthesis(text)

from IPython.display import Audio
Audio(wav_text["x"], rate=wav["sampling_rate"])



