# DirectionalBlur_s

DirectionalBlur with CUDA for AviUtl  

## はじめに
おはこんにちばんわ、作者のSEED264です。  
こちらはAviUtlでCUDAやってみようぜっていう唐突でありながらも、  
なんとなく誰かが夢に見ていたかもしれない試みから生まれた動作確認用のサンプルです。  
私もCUDAを触ることすら初めてで、壁にぶち当たりながらも何とか動作確認まで持ってくることができました。  
もし動作しなかったら遠慮なく私まで動かねぇぞ馬鹿野郎と文句を投げに来てください。

## 導入
導入方法に関しては普通のスクリプトと同じです。  
同梱してある.anmと.dllファイルをスクリプトフォルダにぶち込んでください。  
初めてCUDAスクリプトを使う人は同梱してあるcudart32_75.dllをaviutl.exeと同じ階層のフォルダにぶん投げてください。


## 使い方
基本的な使い方はAviUtlの拡張編集純正の方向ブラーと同じです。  
純正は範囲500までですがこちらは1000まで打ち込めます。  
Gaussian Modeは名前の通り方向ブラーでガウシアンが使えるモードです。  
範囲固定は元の画像サイズの範囲内で方向ブラーを描画します。  
境界ミラーは画像の範囲外を名前の通りミラーにします。  
境界ミラー使用時は強制的に範囲固定となります。  
高速化モードは方向ブラーのサンプル数を 範囲/値 にします。  
高速化モードは1未満を打ち込むと1となり、上限はありませんが、
上げれば上げるほどスカスカになります。  
2程度なら見た目に大きな変化は見られないので、デフォルトは2にしてあります。

## DeviceQuery
一緒に入っているdeviceQuery.exeは、PCに積んでるCUDA対応GPUの情報を見ることができます。  
CUDAのサンプルコードをビルドしただけなのでどこでも動くかはわかりません。

他に何か知りたいことがありましたら、私まで直接聞いていただければできる限りお答えします。

作者Twitter:https://twitter.com/SEED264
