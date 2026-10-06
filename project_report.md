# Machine Learning ile Çeliklerin Akma Dayanımının Tahmini

**Proje türü:** Makine öğrenmesi ve malzeme mühendisliği  
**Model:** Random Forest Regressor  
**Programlama dili:** Python

## 1. Özet

Bu projede, çeliklerin kimyasal bileşimleri ve mevcut ısıl işlem parametrelerinden yararlanarak akma dayanımlarını tahmin etmek amacıyla bir makine öğrenmesi modeli geliştirilmiştir. Python, pandas ve scikit-learn kullanılmıştır.

Hedef değişkeni ve `grade_id` bilgisi bulunan 1.000 kayıt modelleme aşamasına alınmıştır. Aynı `grade_id` grubunun eğitim ve test kümelerinde birlikte bulunmasını önlemek amacıyla grup tabanlı veri ayrımı uygulanmıştır.

Random Forest modeli test verilerinde 89,34 MPa MAE, 162,16 MPa RMSE ve 0,371 R² elde etmiştir. Baseline ile karşılaştırıldığında MAE azalmıştır. Bununla birlikte, bazı yüksek dayanımlı örneklerde büyük hatalar görülmüştür. Sonuçlar başlangıç niteliğindedir ve bağımsız doğrulama gerektirir.

## 2. Giriş ve amaç

Akma dayanımı, malzemenin plastik deformasyon göstermeye başladığı gerilme düzeyini tanımlayan önemli bir mekanik özelliktir. Çeliğin bileşimi ve ısıl işlem koşulları mekanik özellikleriyle ilişkili olabilir.

Bu çalışmanın amacı, mevcut kimyasal bileşim ve ısıl işlem verilerinden yararlanarak akma dayanımını tahmin eden bir regresyon modeli kurmak ve model performansını basit bir referans yöntemle karşılaştırmaktır.

## 3. Veri seti ve değişkenler

Kullanılan dosya `data/steelbench_core_open.csv` dosyasıdır. Veri setinde başlangıçta 1.360 kayıt bulunmakta; hedef değeri (`yield_strength`) ve grup bilgisi (`grade_id`) kullanılabilen 1.000 kayıt modelleme aşamasına alınmaktadır.

Girdiler:
- Kimyasal bileşim: `C`, `Mn`, `Cr`, `Mo`, `Ni`, `Si`, `V`, `Cu`, `Al`
- Isıl işlem parametreleri: `austenitize_T`, `temper_T`

Hedef değişken: `yield_strength`.

Veri kaydı: [SteelBench v1.0 — Zenodo](https://zenodo.org/records/18530558). Hedef değişkenin tanımı ve birimi veri setinin kartı/dokümantasyonu ile doğrulanmalıdır. Kod çıktılarında sonuçlar MPa olarak raporlanmıştır.

Bazı girdilerde eksik değerler bulunmaktadır. Veri kaynakları arasında özelliklerin eksiklik oranları da farklılaşmaktadır.

## 4. Yöntem

### 4.1. Ön işleme

Hedef değeri veya `grade_id` değeri eksik kayıtlar çıkarılmıştır. Sayısal özelliklerdeki eksik değerler, yalnızca eğitim verisinden öğrenilen medyanlarla doldurulmuştur. Eksiklik göstergeleri de modele eklenmiştir.

### 4.2. Eğitim/test ayrımı

`GroupShuffleSplit`, `test_size=0.20` ve `random_state=42` ile uygulanmıştır. Gruplama değişkeni `grade_id` olarak belirlenmiştir.

| Ölçüt | Eğitim | Test |
|---|---:|---:|
| Kayıt sayısı | 796 | 204 |
| Farklı `grade_id` sayısı | 446 | 112 |

Eğitim ve test kümeleri arasında ortak `grade_id` sayısı sıfırdır. Bu ayrım, aynı sınıfın iki kümeye birden düşmesi riskini azaltır; ancak veri kaynakları arasındaki tüm farklılıkları ortadan kaldırmaz.

### 4.3. Model ve baseline

Random Forest Regressor şu ayarlarla eğitilmiştir: `n_estimators=300`, `min_samples_leaf=2`, `random_state=42`.

Baseline model, her test kaydı için eğitim hedef değerlerinin ortalamasını tahmin etmiştir.

## 5. Performans sonuçları

| Ölçüt | Baseline | Random Forest |
|---|---:|---:|
| MAE (MPa) | 143,42 | 89,34 |
| RMSE (MPa) | 204,62 | 162,16 |
| R² | -0,001 | 0,371 |

MAE, tahminlerin gerçek değerlerden mutlak uzaklığını ölçer. RMSE büyük hatalara daha fazla ağırlık verir. R² ise test kümesindeki hedef değişken değişkenliğinin ne ölçüde açıklandığını ifade eder.

Random Forest'ın MAE değeri baseline'a göre yaklaşık %37,7; RMSE değeri ise yaklaşık %20,8 daha düşüktür. R² değerinin 0,371 olması, modelin tahmin performansının hâlâ sınırlı olduğunu göstermektedir.

## 6. Hata analizi

| Mutlak hata aralığı | Kayıt sayısı | Test kümesindeki oran |
|---|---:|---:|
| 50 MPa veya daha az | 111 | %54,4 |
| 50–100 MPa | 45 | %22,1 |
| 100–200 MPa | 27 | %13,2 |
| 200 MPa üzeri | 21 | %10,3 |

Ortalama işaretli hata (tahmin − gerçek) -25,41 MPa, medyan mutlak hata 43,81 MPa'dır. Ortalama işaretli hatanın negatif olması, test kümesinde genel bir düşük tahmin eğilimine işaret eder; bu, her malzeme grubu için geçerli olmak zorunda değildir.

Örneğin gerçek değeri 1.330 MPa olan bir test kaydı 361,21 MPa tahmin edilmiştir. Bu örnekte mutlak hata 968,79 MPa'dır. Bu büyüklükteki hatalar, özellikle yüksek dayanımlı örneklerde dikkatli yorum yapılması gerektiğini gösterir.

## 7. Veri kaynağına göre performans

| Veri kaynağı | Test kaydı | MAE (MPa) | RMSE (MPa) |
|---|---:|---:|---:|
| EMK_spec | 139 | 57,98 | 104,33 |
| Kaggle_Carbon_Steel | 38 | 132,34 | 189,63 |
| Kaggle_Stainless_Steel | 27 | 190,21 | 303,36 |

Test performansı kaynaklara göre değişmektedir. Bunun olası nedenleri arasında malzeme gruplarının dağılımı ve özelliklerdeki eksiklikler yer alabilir; mevcut analiz bu nedenleri tek başına kanıtlamaz.

## 8. Özellik önemleri

| Özellik | Önem |
|---|---:|
| Cr | 0,2278 |
| C | 0,1816 |
| Mo | 0,1264 |
| Ni | 0,1251 |
| Mn | 0,1065 |
| Si | 0,0776 |
| V | 0,0556 |

Bu modelde Cr ve C en yüksek özellik önemlerine sahiptir. Özellik önemleri, modelin tahminlerinde hangi girdilere daha fazla dayandığını gösterir; nedensellik veya fiziksel etki sıralamasını kanıtlamaz.

## 9. Sınırlılıklar

1. Bazı kimyasal bileşim ve ısıl işlem değişkenlerinde eksik değerler vardır.
2. Veri kaynaklarına göre test performansı farklılaşmaktadır.
3. Bazı yüksek dayanımlı örneklerde çok büyük hatalar oluşmuştur.
4. Değerlendirme tek bir grup tabanlı eğitim/test ayrımına dayanır.
5. Özellik önemleri nedensel ilişkileri kanıtlamaz.
6. Hedef değişkenin tanımı ve birimi, özgün veri dokümantasyonuyla teyit edilmelidir.
7. Model, deneysel doğrulamanın veya mühendislik standartlarının yerine geçmez.

## 10. Sonuç ve gelecek çalışmalar

Bu çalışmada, çeliklerin kimyasal bileşim ve ısıl işlem parametrelerini kullanarak akma dayanımını tahmin eden bir Random Forest regresyon modeli geliştirilmiştir. Model, test verilerinde 89,34 MPa MAE, 162,16 MPa RMSE ve 0,371 R² değerlerine ulaşmıştır. Baseline karşılaştırması, modelin basit ortalama tahmininden daha iyi sonuç verdiğini göstermiştir.

Bununla birlikte, kaynaklar arası performans farklılıkları ve yüksek dayanımlı örneklerdeki büyük hatalar modelin genellenebilirliği açısından sınırlılıklardır. Gelecek çalışmalarda veri kaynaklarının uyumluluğu incelenebilir, farklı grup tabanlı bölmelerle sonuçların kararlılığı ölçülebilir ve bağımsız deneysel verilerle doğrulama yapılabilir.

Bu proje, malzeme mühendisliği alanında makine öğrenmesi yöntemlerinin uygulanmasına yönelik bir başlangıç çalışmasıdır. Model, bağımsız doğrulama yapılmadan güvenlik açısından kritik mühendislik kararlarında tek başına kullanılmamalıdır.

## Kaynakça

SteelBench v1.0. Zenodo veri seti kaydı: https://zenodo.org/records/18530558. Nihai kaynakça için kayıt üzerindeki önerilen atıf bilgileri ve lisans koşulları kontrol edilmelidir.
