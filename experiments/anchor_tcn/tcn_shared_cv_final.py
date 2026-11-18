#!/usr/bin/env python3
"""Run the original TCN on shared CV folds; evaluate val_00 through val_09 only.

Save in /home/abc2027/hasti. Run with your existing CUDA Python interpreter.
Default: 4 folds x 2 seeds, 5 full epochs, final checkpoint, then nested
postprocessing selection and official evaluation. No installations or fold builds.
All model / postprocess / shared scoring code needed is bundled in this file.
The official release supplies its own evaluate.py and annotate.py, unchanged.

--check-only verifies inputs and the selected reference data without training.
--resume /path/to/this_scripts_previous_run reuses completed fold/seed jobs;
an interrupted job restarts in a new attempt folder. Shared data is read-only.
--epochs changes the common predeclared budget; never select it per held-out fold.
Default checkpoint policy is final, as allowed by the kit's quick guide.
Optional --checkpoint-policy swa --swa-start-epoch 2 averages successful updates
from that epoch onward, using the same fixed policy for every fold and seed.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
import types
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

VERSION = 'shared-first10-1.0'
PAYLOAD_SHA = '0db2221928f3defebbf7551261687c1a7169697f7e3cbc13a40de70fb5044bdd'
PAYLOAD = 'P)h>@6aWAK2mtuFTU}9}=W8%L001S)000*N003cbV`y)3Uvy(`Uo$XYWpHm}Xmc)bdCh%mbK6F;==W6RKVY$yYtV+Emz_9rkUCyJHmPzPm+hRbb9GcGhy+EfNPq!ATeMd9x8Huv6GKVa-E-@*5{m?8dS-fhdipgzfBw_{rtkL8>t?@dK1J*8uxOj(ufK}p`1xO69332ecM@GxP1Th{6_v|n)PAbEWw~D0%|$eHW!>PH&vi3vKljg~Pjz1pQB_VCQQJ(bXkB$twQi@2?CY<-{_4X}HnVcsHdWMD9RP~@X;-g@EP7uJ0IH9MMHRJOeNi{%GWzbj$LaC$_j~KID_7M3#h*9RMceIteDN0Gy{zWtW;vWh#|P1T16`wr`qXr5(XOMX2hXCu`~;nEX3?i|S<i4J6*Rt$o=6n{_PVS3g|3;xh=zsRn^`q&XH}O)AL<MIy>9zq-L=!I@3F1Ak4hL7_I0Sf{;ImF`vIT#L$!X!7@01r>E*hun}K@NN7Ht-E{FPjSw(XkVzln6Sv?)<7Epox1^CcRxx^m6ugV!C2u4u$)c_b?Zl6V;yLtdv0JVi4%xN6e6(GJ|RnXF`?yBj~cH4|5_3N+ZUAu~kV!jzRT~!oOy;`^30L?Y+P*T%hf2E#v7cic_(hsI>GgMc@vOYJD`cL}%qU;y;ySmk1|K7Ka{;?{Dh5pf+_Bt4IyV5`V#b&6N=I7Rw9AI$fM^{d(^KyD=o^02ZDg32g&+BC+Lr3gQmu250(qx=HVm&plOwkHkQsva|l-mCSbSwp77IEOJ+&dcI*Iz-8vl*bQpgAhew*%2olt&-C4b*M>3>ID8wM_=|Ib(WG;$MGxSG@i0m*V43?_WKC`5{iD_;EbhUHsjvw|{x_{6nygnVTo;t^Uz$R_iScJ8ISjXKgoK$awl`4eu4+iQeT6l~Qk-#(tVLje0TPG(^R)OrVk1|KH%^Kdc$1KhcY}nb#L5Uw;+B2bXTK_;M1>mu)#f*Pa|?2kO-fR>BD`GkAW?4}oi%N8dh`hd_uIUAt*!#ZYdF-atE6z?kdkng^=lFkKXV{iRZG9`lQ3Rd&dg3hbW+uG*6VpR4*}F%-a|%B^{SsNeThHB&H=rSsD%FqeY(j_%6S1L@1GUNSE@iH6O3iPMxErqNLvJx-&iX>|B)8Xd#0Zy!&%!lItdsz$*%JQ8HA%B!MU0tW>iSIjmY0kv`RMBxO8tOb6w0-jSlC5}{yc{?qaMY&urO7r$0J9=Wjz9?6#GEjNhP76kpg&_43dwBsfTtF9AfxhA6;v7I+_60E3q93}dT<O7n>zDGd`t4(@003`0gfa#03|uLMie_GQ6>OiyZjzwEuB-C0n88L<Vqv$@Ljg7go=SUO(N7V<)V__sg9AYzzrX0}nKkM7I-Nw1_(@Vn`v4jC+qR2pXbD;NMTMjRPbWeP&8i=PffCs<{Qm_0|4z!y>#J(U-7+1@p4u+~l=bQm*D${P58N25ZrBR7R`UpUxoSPspJ0S_wVVkylD=BbC$hY09apgJHPIoA=yCKg>NhJYl4*jU+|q_@e|NxUfZ_f`d(q(}+TV|kZf&Nk?r4LWuof`DJ;LG8W2cwc&evZd*uT$!>nFHXrBQqScLfz60CI89qL&l%ID_3Es5Dz$f)tVPk3L6!okk?!6zwJblsZF&?7sxbD(CtrMHLNngCJ-44G<q^!)hJ-bu!)<u<Hp*3mJB|PoSqM5E?L-AqR#7B53;=KsNbn5R0k_bfkfO1h``!Z-)8aQw0|`h^|^MK{7~m6KNy^R4CylL4*;sbRfj05lvX1zlHt%e*j4g9e~LfSV>DLm0;Va2WL`><WB+*)X^mIXV}G>>hrR0syvPZqsEPP0c#yNilG)7Uk7c$5VMyc<NS>tBwSENinG$^+#BpNh*Nz@+jkx_S5`PRAbGMDAd&*K9dB_!G%5?>bzcmCiUb7ljYh7b>7kS78GNnR$s~ihM)=5Ru4Al-d-);$v#omigZ~WijnzWa*Rx9IlV?M<Gjv-EH)bw@Q2KWHsY>u0vH@7MLz_qef?$eFMr)Df7kp4%;hK;B69@6C>wwRW3KP|u-r~+&kTNnXU^Wx{t<gf#aE6cgxp3EYw&M*;VXLy)z~11$tmpE6x|x+3%tQI91fB=`a$@rrtBeH*cHN~7%#=iFr)p+BR%x2-9O!ZZd-x^LoDB8W@-tNHR-ki%vNYAQpu!GpZBx`Yfr?rX&-?YV){{AJmop-5d<VNy%$sr_!*?pEih&>orLJ$(?~WHUkW>$%Dj<m=Wf#l#Jc%D>izo5K#z?=J!(OB~8mMzCAchrm3zRi@kx0WuJaL~>Cy?6}_VNMZ#HAl!)9ah!n!(@1rU=XwmVypa=dG?VRn&cDkytX1U&1b1Pf@Mqu0%C*%NkZRZ1mRv>&ET@0X>DMXEG*DAD{q^n&jqQgP`)Z9bV&0)#}?e0$fRxJuOkjq;5?}o<KSabPXD*2vM-Z@!vF>&M)*z1@#J^z#iIm#4E3F(&+TeOd)hXHS@?2GT=D?&z?KtqNw>~I09h`akqn8()=vSHA*b}@-&{-U3`|&Q@0@CYYB{*-b|uDfjEwwoPGiq#|JQ;jMWD7=19LG-N6O9V|r-@Pe`Yq0=-7bar!X@bz?#Zs(@R~V3rZQGtX*L*Ra_8&G`&<;=~;mJxj3&S|XO!6-ipahbB9F_7Q4u$KqG2LuiwjE`UW;APYX|p;w^yvC#J7=lAdvRzwdJNdlJNMHK9y1F+?EUAfgA9OARnlQU4sv5PqrAb48cum(6}AmFYLt%tOi!lVo|A&l|EGwbOO(a|5mInm)zf*eBT7#s}_5{L>|Q+d%<RS4s-j0o#<;A;bjACnB(=LGwKBW<<`{F&ABIdutS@ZIPRb*gU2jAeB$(24TAPX-hxq90Ka{FWN$jzMRp%eJqqUV!)xY?BF=RA3OahIW}_O`tOQ`uLCav$E-9y*URdX92%9O${q}1-f<g(-#^93N^5Z%Z?0YKs8@Lb{NqKj#ja)!5eQgZw-Skhs3KZd?YK^AWO{>H1-JFDe76Sc}b2E2uTDf6HiWp887j4JaqqqHjO^V&Ok3nYfGaQ_Pc1zWA)%3@Ql8k3;jBCG;!Ii%QFqXg>^$B21uo}kE%%|-k5m1rxeUUUWEOOL{CrB-0SNLM-~n9Qk45(N(g~U_<YaeB9~RYM{W*<Ntc8=E(qWQ1rbJkma5+{gE7(oF@(e|1QX{xrm_=fYxP&>>21xN$<n=R4~EUK*<*mpkP4JexU}h*K+l)u#YM$qNR&JbZ?cPONR+@0yA((vuxM|il2PRbB(%Qhs!w(G`F<df;&t`kLENi{t<)R|;dvg#@ZH<pancO2op0(T+Pyje2n<L_lC&yodynJzBU;+Yf}KUbl>KESXo=9O+CKx_0d`S3MLT<qB6%MT<t3TOr_~ItBN>^q5uUUM%8QtqJc;3aTd>@PoaqD}D;WZ*jyidB7cm0m(Clfp0rLl?7_vtm{P5H-f2Q~kys(I$*w3JQ5|Co1AVA?Z$xOgJ2Yr@yf)w7&K|b!#yoFQ+^o=aEPpGE9i5?%_hne9b!KwIHVf<S<ayt+UB^Par^Q<Ob;aKKWCI&YVZ^{&_C4ItTGkYEva%;7y<{(4y7B<QX2Ow-{s6VjGx5FAb+PvG#z($Kh8&i<_RG@1GDzP~=;^sXQtEOqwafA|So45TbSxoDz60|r`oQBlH4xo#Hr%2VXIUrs78##W?Z-vIFf8@a;>J9;;HE>#<oFhj)2u?iW&chB4sBVVj&Oy0j@!CJX6ABH+8$&=`i+;?b^S}h6F|Temp7OC(!*#}f=oZ2bt5?uJAijFFStSR+ZTPWI+`iO=oecx<)(DpxQ6v-kZBo~~?)o7wfq|9q_Z+RG_u&X=I<LBX2jV@M00_yE20+>YmSN9<Y#&Ovq>gTL#a(lY$L1EJ&FNobT|qve#ynD>U0K2GQ`%L{a@RBV5i33Z=a;N^&`a7OdvEt0?R;h{hK&-9S4FidEgidr28vnT`TOw;rn7d_qtAX%Ixw%7faqr^oP%Uj_aOG6*B;fm0R?rmKu71O-<MqtdzapkQHt&@v&(rk;DSTPZw7invWNOcVhOX+L)MBt=h?*HDEoHPO{?3r{#k{tcg4^3kfBA<O|oXajf3U?s4-XO-iAQM`W-SOnjv0M0i`UxujU&HAQ)P}>juh{+~tTot7Q`bc`s|k`ldOFuC1mVb${LAa+t@bGVjr=U!I?d^#`TmO~3FvZky}v5s=lM?NUjM7#ZqvNjS4Dy{>76=j9IL%Czo$(~8HG(FVZWW?7$Ss3(>KsUgLxTCK{pZOdP^$WlK)ipL^>(%+y7w%vHViTI9dcz~j(CuwwWa+pT<Fg);z$fG@??1v1|YXoT)d64mw0qfLVIz5C|9v^)FZAh!`2Uu9JUg{=sD=@%ETrse*qWdv8FsgWhaPKufA#7h;BKTpY$R%{gAYf@nXhK0#p>LJcwJUf7dPu4<%;5*wl6g^biU>~`?=XzhL-U<FmqxHn^*KELQs)FC&<bH@j1`prq<ugi8O$s(GeD{ttt;Q$FsqjQ+C2Vq^q*B%&$k??v@W-(m5VJYB`mM1u3B!%R`T}Ue?{zC$-MPuoMr44>dtRjZNRB}W`XW<0=78=A@HJ}DplOKmXvH+PXhI5Tm|>8I4lm3NOVISBybQwIDbgIybI`2;R7J*PoqMJmS)MeBzSU%GCL_wlMNM_-yZSyv+mlNGTT06h`fi?Un+;G8)UOhNf~v}wx}8jGWnl>RfPX@!-iELWHgv;btv78c|7oQ-HNju!rn`ZN!y$s_gXEym?Dm#K!sR1;J}v;1G?Sy{{O!G9R2k2^%Jxcsx?4_SWYp*?O9YVKbPA+Lb~hKE===A_TJeZsq%XNM**ZLaLE@1`LNl-)-%_3+$-p^mUr}h#Zf_4aj!7XswyshV1@tsM!D6o18CIXwqZOISD8oUAD3O-*kHWr#B!!fIc#p_=QzZzt!}A_u(n1$LVCx0<PIR*4FzKaH@Io;Q2he%aOnZx8TSac)`IubE>=$Vbv?aYR^n&31X?Qs!2)k1o*1z~y|CF-_?<lyS89|rSGDb5lr2zgcwE=KUbbnps7a>0!hMy>O;Xx2ubd6DUAw{>)B=5c+1_c?dtd#&f#rmY73JIo#>=(h-JsYCdl~^1?!J44`wO(Ps6!jBv!5WH(J+6A4#-!8L;o={!|-)i)z%s&6MlfsgD5zF;U|KQJ&g~A9ss##ft+Arsx(oeR5iPB{28d1c1HL>z9-%f<4n+!0hGL_u@u=NdoWNKAk>B=0--a2Ym*jxF8rt%ix{&hhRfMoGRk4Ri1ccgHI-4g!B|$8eS_PYgxQa2M=4x$V0IYd1w;En3>efd;IR^qT!=9gw45|*Hzp{d9e|#+;iBKHiYuO6HhLw-jhJf3;bf=$hV}d3Eda%`z669JSSd1X+ir$?L}hDeo+;9FIp|iMQ8)~)H8x6e*G3U=&T<BN?3FOm^lRMP4pQZr+hd?U96R1_l96Hv8?ne{pzv4?Si{o%liJLNmfR2?NodiWOL-(fj0vYN6J~~}K@Cf-E9&mV7Zb;N0?p^699Q0Em9T6YC<Al$h;3rOZ$NuXt|q(7Gs!@4>I1)9RV7M#+}u7F5sDaQVr~1qBYTOm<4Cho7;Se5<01Vaz958w0fVju2sw`qU<8cJtZDP(8<jkcKE`VAsUp*9NmhEhI|znqBjd#2D`ipQ4yh{mD<GXqft_+YBB1<KeMw;Lp~K5rsi53`fqf4_=T>Vu{VM=Cd@fWQbY<8;K~BW5HPlAYfMl`JX1y7BcV&vguM*9Y$F$O*QeUE=sNw{P>}l2KG&`16OcmsH#>AUj#4kw9>0eFz762Sz0ZTV$R~g|GS2Aq`^sc;e-oXwsu>lZPEC`Hv1w!6x81j~YcrHr|)wXE*UgXb@FW&xz7GKk3zqGT>QVWg&@p1}$xdBlq)v?QN5YHNYSedTmbM<N4d^$veY_b>lb(R=_=G!rfu`}|9s#Z*U`jHPCiA7-l6<FB%E&NWN8n~gp9Q_aW<>>al7!(7W{9JZ3!SI!hP_)v##&Ee!V#b&lR00FDO{`lb=TZBn+)=U(MAQjj;G5BQVp9g6{{gzVA4$wF{tyk&dMj4THRI<4ihm8-zOc17b%Xo8+JRGyaR+LLH}9)Aza{)t3c)K1jkjL=ZOzfrmq#vOi57YYb(y1bRDGx}Ng4v6+*TdU<L)ZFnR0o?+Z$vHr=z82KmajKXrm-ADL&UR(q{pRos<6VQmq9GBgFk$7`c9Xts}P<6vC;?nEMuq+jhqB{ldmc=AGpjzaD@9l`v|47{NHNOVm2iP_S$-lI#Et)!FyoPwobyhvNfLR&Q7!hqNm0wZbU&KaG^=$NNG`XHKGA3G$}+7q!AP6@wathrys!La>0YG=bUBCKI>ZxJ^f@!G^xz{=eD`B*7ig>>iJxsM;1)dtht^qKVr4OuhiNGT0zNT0}`451@EL9NRmUSkF}MNpjc_%Wz+IXt8?sp#;ibRrLiLgVp;w>)ZLTUT%7ChOuYkj_I(~YN#lFx&Q^DPk;pKf%c^yh%@N4cX-vhYtPH`8bdKK)QQ4o`}$(l)-!>n1o?_^2y+O+M*)gd-xt8JK%v?eWW(pux&vE`4w`Kww7UTbOX*u{G=|&lJ_L^aC`t}NDzqM>!Hk}5r5+u|5%02fJjbr?y&8c~;T-@4n<d=H@Q~%#bBt_dGiIWIbH9d3wr)QsrU5$pgAD*;1QJWp=<`$U@B`4^K#W2eCQ%HgMxSEO6Q}Wxp~)zm;@q8PSq5_P8P6P@iP=#G?GCngaML(2Mc};^=BSJhlTHAM0E)OFoH%3H=XUL(R(s!05J3tVe*itQV7y6rMDuG7R{?ChW2sj@b>5m4gN@0tm1I_Zs;B<h0LDtW8K8Qv$`qsm#~7?mCn7LhXvR)1cFJL)j7`RQ#O2rqZO*Y~&1Sh&O*?BCPbae4Cz?AO3!J5joTH{-BtHg+3uhmN$gA=aY8J+wF|V5$o!`Q+76W8KSj-Wr!Gl}(@s6tlVYy7rj)P(o&29LIu@yE$r@NY2b%mUjI%$hhq&KtOjU{{z%{(yCcl>}LV0ZupBJeMJ8?7iXA8E|4wRgA!&Las(7z2X2Mp!?JH^)VhIYNC)9D)Q3<YK3!P=Or&zYn_k+?*&oW-e7f5n8z{yA_`&a#xe?oFP0h_b>72TcY0slm%|D?NS@EFDv5OUR1nwuHky^hO5!h2bxg@^~RPr(Cv<r@98>F5^?N-%aN96mMxm_AQm1rfu3R&d}_B&?Kl#oWf?N{O#H^~ns|_8T#uttcGm(;xl*>;?gEjO>~Y`I=n`fvhhZWsalGA&dRdXJifa1?LtZQ#3Pc6WK`!;d=~5&}T~Y+%JKt6k5t4I9PY+Md{KlM(%8ksyU$P$|cBGM%P<5(-pG~5@`;az%91{)Bl!hrkEvVjL)V^o+1lWNC1(vnfHB-WQW{=u!MKV$_`t9w{>N)j<EKrBOBUh1wZtYY7{J|``GiTY#0>dc<lBu>i?3U-VGFqQRiMvYD$R_2<#0$%A=dd$^f?065E9`Pkh_;+M)G^xK6i+wB<@pp(anpCHSY<k*MZESD3}5#JW<GF?x4TdDI0rgFTV5|~_@Vk3={{oQ&o#o=_ImLLN(esrbSW5Ksn{!0IZ*<_#0;W=;!%<$E@U(TT*OXo(A#k=1#O}#P+ywg_8AxIPjz8D)>ss12Ifax^v}_&t7;0Gi#td&wi?Y+tG3%di)Jlhk`6O+ehn-8hPdV{fR><>*A9ZQ&Jn~|AZ8Z}_Y3!*qOjtc=u|8;o9M~J2yCJc6$;gDh6O%U`AZ5bQAq50HnGdF=kw9Ni)ttb=nc1?Qd?|XorL`~?>7zZx{4%z9Z&xAcRyn!*sA8dC)Wt_<{5g%Ogfmd=QUi-Vk2gpS9&foHPYl1V>sadYaClarQ#DjnSWy<>tax}E;0WB?<LA_%OIgg@%grZnHfs=)rjVuvBkeNI#NKBcvo@8i|GQG!JT7LR!qSy2^N8KBu7ua%_TER#y^EnCfTC8n$;H-8fY!n0J<g~kORh&aEGiDp_jZh@bgm-GiS=;&k~xv<lugKSX4OnVg<xgHn8%v2a_a}-mf!9*Dp&iY>{3-zrE3m@Fw+6<N0BM_7^&fu_rItXJXu(7)S^cg=EO`I9W*QX70y>D*)UDf*?KJ4Z=dCqOwBEF^Q(*G~&gT>L_j4{w;aS_tU}Gdq)KVlfisHZ-4oKu@svnjTo84hNuk_BVkx45CbgZ=zJ^bh>Y-PHXdO!m{NhG0&`e#Cp)FYhD~;jGTEtA3^&H)#M%TI=E*w=MQm1Jryz2LLEVQ__;#iwLJ{wN%o(sHt7sWl156DYF&scmgHuk=iDNB82TsB)%qi`z5^LvVZ_PYrCROyCb7EVl6!)L32v9!{fCJs$(DAgn?B?_pWli`ZSg<6EKS#gyl_Ze5D4Y6A)jjAV4tk==r+`2X_wA$e3f@$5c8d<dHCrJaD+G)RQ(%)pV`xW^LD)D{AU<IfTJ6CYO@%ZR22nc<N&(~(m%A9(ZG8BHo|K51-@{`^<d4APKs9Ivnmsd`t-jbkIX%l-kl;Qjubl4bh(gsHMQWUS1QD9@_b4~3hRjVSaS7Y$b+VF)1p#Xz!N&9}PRDR3nzS){k-U*#cAJznMVYY~rj&})aBWrgb5*TVAg`WPMc(=?(Sr|e+nmChF;Xe952}onA;wp-N2}59zr|@h&o}nZRm`c5W`DV<YCVKscWUNVm+17Q_51EqN3VGk%Xe(E)-HD+PzO&mAT6l?jDze6gSsb_?{+}>PC*HrE1{W(&4>0$oX*h-MYD-{h~;6p^a=n5!pek}%w$<pL=z+PFP(LGX>LyfgLWI#OJ>+Kv}E|TE$yDAUL65sSh%?8-TZt7lNz%9l*umS)@MxM{t$NM)2hv_v5&X`V!`DI*4)yl?CLxcp(?J-kKh|hP?pcB9jZM3hyWr!FCd&B`6kHvAOk%^i)XcL*Od2<caNTfix<^$wufUJaP}}nbcLY;S@a6ed>RG|B+$b@)tYuLj*<sm2hU|^v1hxY5%OeMF;KL@6=eMkCV7dQeGt<zXc%8=(j*ZMI5<5$$&Tlk5JStuC?^#%pu~=yqSG&WpO4?a`taMEj~|MUzZM^Udj9@Z_N89SE>Gdpk_a&S-}RazGr2zO-*{RU<+HA$jVRXze|}fIeD(Ux^N+7yQlK@SG^#&U9+UJEilC_2Nji~uq-O>7^GHF-b2TbdY(NWPkt-&xMzRf!f|c+SJ}8Dl8l8%#(;RL-9Yy1@tco~WkuZ}DS0vq_?~?wmxA9G&MZpCp`w^J~p5Z2}H6SOU4$ho^Xr0DtyZAp=)h!-<-~cV^TF|vo{GWbFzn}y<rmPCKB*}edpbnY(&}4L6#w!9a?|QAz2Usp&dK8RcRBet<6@=g^U2)i~3wCt?OP7GvRBRY%>Yv)NFkDc(qWpSB=MFkg#)gLbq!gm*4hViPetY>G8NjB|^(gdk0ngr19@<D!JR(EOv_Q?NV?mnMIaUc$pi1lr<W!3Bf<_FCTmOAicNLwny22DX^&pnqwsBi}xBW;1wZXmTj8M@z9=3XbD0rac_;ty)5gLl19J8~6P*0tJ)-6!Vcq}Lv$IION?d!W1=BJwuHBv<d)}X`FtvqG~***_uKmu8j!Y><W;gM6~{dQqhOOJs5gaowS#3PH09c_)%ySFDj#!jU4@5x)_gjn0<99+=PJiv*Fixp^Gb2O`n>rNk&T;M?;_vGWNX0zh-#?RlpiBPx|XDDjGv~6s9qD_h)`UKscNoYf+mi0%V6{<+nJ+B+m@`;2xN^A_jkG?<J<0QYh+dt!|K;`>5dh$3@>G}wReB>{S{>F)w(c+4!9@N25jMRpOBMYJ;e+6e(k&rML0?_kTsO-5UXy)NGZ6pa~lx>UU2kd=+S%WY>BMp^~+oNq%*2YqS)2BQXn(a(YY|z`~lcM=>_#Pgpk<<n{!;7u3QcrW|jP5DhLDA{}YLTV%n^1zpEx(nrljwih5eB*ejNoKf3b0_zq=k``Npgygd3?B8p<7(yHgx25-C(@0Ep@9N+z*~AeJjXvxhy2{fts8=!nl3QE`Sb;@}jcC#yR_p4foiOof)4^<VvtceD<w7fl<faF>~X_<?8Cd2Klf}CQDB@VYI8v7U(zA6LAU<(c@zt5J=xsCOeP?D0B!$P&V#1BQ7K}DXMn$Qi{4ohcS|>Dl$nU*OMG-15m)mS3-pHF>_S#=!HbgxH&unV0qa5I=65TD5{B4N+YsuLRI*7o(al*l~R!!zdxaxE^c935?zWqMy}$62_?R|m($@x3tZA;QC%fDaZv`ZKxUveFs(5i93iUBui+o>49T5|pOCbXcNd!J=Vr;do&CIMm&+||F8yU?l`n+Ia`|f#(93pYLEKYiJd&%Mf+tarvrtH+`SLI-!PcTj)icJer0?)ymsfufHic@3VNZ;NNQYKs9o)9DQ>H+|2<@VR5<ZpFt|Mp8B#>U&7Zy6dZB83E{_bH)n;AXuQkK~g6|OCj1q1-CDw8mGR9c+ATBOM_p2{>5uBvwTf>!xmAPh6KZ2~^6xcnusx2S@+g-~V-Z-LWZcVMSK34yaRE!+bjuHd^M`a}kDL)&jUIm|#g&rK6H|B`5pTR`g$-4ptF0CzNiw=^&$JfJP$$fY!`>Sf~0iA<--&20lR)tTh<ofLSAqtKH*>q>vBB?E?3u%&XRCN&H6eDBWff-Ctma^aEg3basYP_F2j?R2*k49pGG**A6WK`OBua$!VG0b<(>UIWruQhv$K`P~)-btKQz3#^kUsnwQ$f&kTUfk58{41Z@>>fD&$h6hC&B!Jv1KQ_Z8gTVYkPCSJ>XWJEu0ti9I8ORt;$x7`e9HWGFR9Xo-4vh9QwWngt6^v6l5vOaMnQq$>zB&LL<)a*Gl1gYK4B2O#Q8!Ku>g@x0B1?G;8c!2yB&)+@cnB9)he5gh#?i*>DTT37&wTNX@(EGo7zxe0g<Q@<O~L{RxD53NGY>&^XS!4(z7@zNmD6Bd$|!Gfi?UObWUD}?fQmn7or+YWQ)%@KhO*w`Q5Je5N|yf?_=z<(1K5p!B?y!<YZrLVg*AF>_4M@rSPh+#mN*|q4>>B-e|j)l09QL=@129|$Tt0j;OMJ#7#7uBeQLd)nA;Fcov|PMl+7!c)u*8QS15PY0NZd<P8ddsFuv65<RLXPIkQ3;kS8kUFPfHN@-URfwrTu|2~`)H`Fshq#CI~`O-yRp@_MeD>2fn;I~5xH=~5$3_QRk}ItLvi*IdZS4>{nB8;qsv@6af%-N-&dh8gE2$Wi5CVb#%;w=lp*lCAwUkw`3w)nJ1V#cZU}dWkGk$A6fcm-r#4T+g4V?)Yig$=PYro#TN`Jfm%npW()k4Xp}R=VEkERnL24vd+-1gn37F00n``^lessLPGXB6FX#2caRtXTfqG($BVFzkH5b?o@s6eG?8f@520-EYK?1#bQ77T^(_f&fPldX+D1rEvgfmM^|xf)VOTjG$CcZ6(xlD}H1ytnV4wLl4Z%{${t5&BK#vnYj6Z0ARwh7-fij8kl?%KZPaWqTWrFdnQxeUE9Re&OUC&ubGglf20nr&>);pry<VU=RncOm|QqnYz#24!+rj_Yk5<*&`&N_Fi82Jqr$DzH3c!p?cxc-8wv8_S4amZ@1QwMBdK|Hmp(ON7xA`MfRxIt@<r5I&WJl&-Eq%!3Ib{yjrGcL9N@5D5%A0<hOz00)`xnP;yU~q%sxrnn$S#uMYW4ah{MP%llB2DCn0f!yp^VDMQL0iU-g&7;`Yl%Jwpa7&gG{r)HOY7=7KP!yiYBVxp*%Ov+)))w6#IatTg^z!zpT@|pBqrBuX4l4u%cZoKoY|sO2&`<EW~1~X)!hk2LlMf0$<{laMHwbW-Lhx@k!2#ht&V1AMEAr`6I^veTi|gwKLH+o$Rxh|{E_{!5Aty+EAaQL$+%#dQex16AusT3Y$^@4j4N%55q=>2lCLOm#1I8qOvK4OG{0^+au_tJGSkwTU9g4SwEkMI40wmTYarnp)~1C?T<x0v+=xgJUE>lS<$>dkf`xQ$o$^A`LD_K8O~Y9S5={VJIi!RdePnWMT3uL|;SGbB%cWkM8^DU-#-|ea>SAN$jeK5o;tYK$0kk?@`a}42^69<%O%HB#4?w%bxD-yCA(#S*l^^ZEBu>aqEGld>qw!vr5=tLx3d2KnXWVMchV2yB7C&)|E<Z&o?K@HhULI2d|Cwad^=8Bh%oc^*1FAl>oASnGZ(&e+XF-%E9Ah#Yis)aV>6~{Z4F7=k`4NtaE<(cXJ7A%gt%!Qj=T_?}EuVb(DdbpiPFAY(XUh><+%uQC!rydMZrp^XgMz2LkU7rJ+jg1AkCBgn!n~_>^i6d1<bW)P0peEaO7XkH8uqV^wBzxK{c9BKPalMgbq~&Vra2X^matF+d#H6Ehcok>-%5)K_XuHlMLpO4w-VH;A)1%<5*-sgx-tAKazRq=s;FJngLRNVdg*PK&>b4@{Vi-TAbZ!;0PbaOyT@>UQTih}jM+il>#I=Aow;;}7@s6k!xv+OJz|jzU6a4Fpkzhh6;w@bPgROX%YNTzTR4{>V-$R03Bf05r5rhib|Vs!_NROX2nDbNV_0qrjMwL2|B&z{2Rs}}%sSTLJ7X}v4z6j=YehrKEmpPfP^4E`(}Cms3ro#+dIpsrB_NSUd$!c>&zNR_&ivVDCH+WDZM((@CXO|eYvEKCCkT__szs%u)VOV1ThJV2H~&EIpoPzq6ka<^JiSp1GmT=OOU+~AzSm}c_8n;1llg5-hr9KSWpH3OidOOR+|umEO2=KgrTW-mJKJWu=vq8(;Odp?V+<h$V%waMV5Bn+PY#Y}H}P#1<s^(g`{L(_^lGr>m~UB5#$u;i@|62RUMEJ>&e9(5(DEA3PtInK)bYtMX9*xN<c2fc!E@Y@qv<qu_HE^4QkRE>HdK)cXah4+K|0!Kbql+8;QKU&N{zUir9}iv*H!#$fHg91=ON0zvKKI;cogvG=r`i$QN2RjBKnD~%hD;3G<v0hNEUh;6CbhBYM(CyQtlLtC?^6Tu|B`PJo)4aKQ!jc^pmmDQRZkmt0}rB=m0kXT+lHAr4QSw<J2>8+tAsri`*sk06f`%aNo|^h^aCN&m0n-r2bk4&WX1uWb4r-8O&YB#mzjr`Rm3j%mYXq?OrAw{H(WVn8Uy){tFl{*n%hKI}{8C(kN2Ps7LC9ddxG4k2>t{5YRu60Sa7;BRW+Imiunc4Ps8n7<%6Z*e=rvqQdWqLXl^BX7ZDr%%8q{V4W=V%TQPJeV~Fz#(1Dt_T8ZhyF4v{!m@fCoXgw=N&`~$3|PUcJCekEp{DK(DL8>vcGx>DjN7)_M?vZohQtKy!C02-zM8SOJxtqoFg8z!RyhSUM7M=~kpv#{T^M5H74geoMq1uEMQ(>#5wvuD?SS3?_TVHtoZnzr(KCJ8V`k-+&=p5A4}RwHwbyYQVubmytkWkU3Fwbm#&=3B|HS_8#9Z9Er(0e5XEyg;d%0yz|95Wc|Mt!N->`{W&@ARPE|=e0l{_LBz*jKqIL6A(ii`R@Pn?^GuawV37t&T~ypaCr@xzD5ckuOt@o}nGB$QL=I$7k&N>A=mbxa$*rp7>@MITmp%_&8$p;dF*RX|KiPmP|xeQ9~r@Cs}`J*y8Q{67##h)e!4p~Mo*(~$r&Nu#iI;pl9;L<x5%LocevkMybI?BkNGxT`3}W2QH8%2)4o0E1xG1<?Uzvctmx$_3dMcoBi4kc7Rr);nsXun??zaT?Qmea6rve85deJ4QqZ@IsULuH!kSuP*U|q?r-`R0NTgTg(aZvE11v(HQCWa^iTnvP0%4=_tG~;AYVH<+H_@9~tO6QJ3l+v+UJiH+E#I*Lj-wx_W(~jp>@Q1hOHB&E1VMI^p0QpBa*PJD=lKQg+AynSiM&8JGbPjw7JH(ZNl&K6COQhI}FoJlYA#b6F`fT^^k#qi~~(eh+XTj5m@{BZ?=IKqHAUl1DV$i(Yb7QfEg^|7)J8-SH(dsEHl25D>2JPCN<20R)`D|Booy!-PphajlHo4S-g;=V&V@-;@LAK45fGy41v(7;mq_X$V{9gaXqlRi}mnJ>?Z*$G6Klj^$?66pYWr!W8GRO0g*4xp0=M!H!>f;nVj03Eb}Z3WZ>(hq^!kP@*H@BTe0Qu)Ke>g1WVpS~HA_e-|JmBxSdfb5g8-3R6?(y3=T!Hv~jek<-)(b$Q|iaT0UIRCuZlC%W5-!_L`q9gfX*4$O-scqHsG0RJ(D9NO|Cm$IHXu!B*%I*QeOj*v%Dy1J|_EjmpEr9`eXdWB8=Q>h6seZb+4^Z|!D2>}&5-cjs0kgMR2<~I2F4&e)J0RGnS_O<(LPS4)D&)mTTvl3aQgGq~Q0FwU5&WO~T`a2RO1pvUh9#=OHoysb+K9U8fX+O8WnJV-O=z8Nc63&D%PJ}TM<j3?WZ%B?&E1s?N)}eZppM)c(C{=-c$2CH~HkjT+*`xG5;=Z{b3zS_A0$@Q9#pG|G@5<O<Rix&nqUnmsmlBbYkX0^=S>l!Cw0Fn<h+Y^2;y<EStXacTyl#d>eEjtnUZ(Kv^<k79%HL!8`vm^}f_GxTkAK7=Sv&s6-rgSme{#n^v40_P;EHKI0Oeqr!zix1(+5Dw56*5P{$a5y^DxAxthXLl;x|FnUv)9Ydc-vzoSkHk=L!^Yi1DK*qen>7`eEQ&9~vk>HIw$|=+~~kpg>eDte=3e1Cn8<c`IE{ZnJ~<14!<h{v^77@Z#5B-o1JC@znz|N2#mVNd0^8?)m$VKR<u-;3i7Q2V~Ksc$O87-&pEwuPu4AN#-{l>+XK{-Q)E5`1?I=qF|ZqS@iVaK#u*=R@OhW4g%}5!ML*b&GrquVdgethx6)T;VE-cP{p7~@?)!d<lIn_N(7NnPWgZ@=dX`cFlU6Kl-u;rXbB)rHdJ=Xky%@@k&hEK8H<v-zN#MR6!SKVo-db?&M=@tF#*qGbpF{Q08YT%sSHe3ZnT5!DR2u+u7;Zz!H6C*Kd|QXUa-x8_Ou3JwM4DBEm+LRZ{dSV9)kz7t*ljatqxz|3ycI$as&6rzNhfdG5+%e)B9G9sWQ~n-g$?Km)+U;NGHm1yX_@>yTbfB)4HcghHoG~rm)ZMzlQ(b(k!#FFrqz)4w6@xn-2Rhf{mBYc9oMxMw0rfgb3xzp#oH`Hgq6f=N6)lP|g{;!(q?MIO8;lxnnE?IT;P~QF3gLk_{=6-jivwT|<2f;y{<#%pMb}@F7ukSux$CI2&WK$BS~2t~n=nWMvFOU4!Wx2sK*3!px$dkr~Kcya(C$hmQ|cb}TMgf_w{|A0pnzC|M@rWgx_=&ND*Y&afSdO~DMK<#d3^ANok_9yFVpOYCzqL(PXOb>xMl#2BU^;LxxNYgpNO>$d!wLOQJy!eDqCPARBsYo;EQ%P&BuZfYelBdNRYhn6YKn%B`nf6tphC|{2~mwfv*GfREfBpRmO1RYcgC%n3%B{AcxlZOQmdyfV?ovMa9B3p6dCJn;mHxo|hQjqo?|92B-e{b=MGu)ssL<CeVx^E}m4D-FGu_a9bl}Iqs6>#JlmrjDorBk%!SQ(2jm2ZwOGW>>@f<WJ41>wCk9VZk?Tw7r^pf*o3UpeauXeK{`1t|wIXjZC%IvivYU+QmAcHZW<CL)#==;#IC$Vq8h@mmgRE~%bWGL?(0CaL!r%Inj;a^mp%;X@nZ!-pECq^V@1m@tAIFN?$cWKvyUW^Sx_)@^k;YP%=qe3YGo!fI3+!iMjx=J!q~MdV6+f!)4+m$|x2Z&1~pi}Y?3DWFFJk<WL(+X&w@mreV*$w8PJrEu4xBg#1L`;2X0$PJQ&C!~mJ3W|6#xq&@S9e?kQAwt@^{RsCf+P>H`Pu7JV*SX38EHE+JhSPiq%!xKTsDqa3n);m6waDYdz9Erx4l}oWMmaaxDo2Tt332;^hhDJBx&tskzB-}2xKLL^I2IU;<G*lmY)_ev-Gwp?OD5q$WtMMdV`Gh-)#hz0D-SVe?VQ|8o#9#n>-rSsP#~<E)V`N-!B=F`+DFfWeE>uq2PK33jwCo1cGX0n#l?-p(S10DcW2xY`kFg!ST3n<g`Vpi{@<U+XwAX1{C@C|l#+;{V3?Q`*^j-B^Bh{|RaI4XJ^0nIcj9*-$A#r$64&V|J~vqh&Z2AQ?6egP<~e?zg@VH*8a1qYAuPUEacVTqKLAFnY6M6L*YY$@M;=!MLSkutdVP7aKD}f*z2wB=#?%Jvz{;Wkzjj#I(y`@itN_P;P6DVl&MN>9Z#zt$#-VuLP;0i84*D0s`}ncK*yP5#)dERd-~D<Putw~g(Gyh>A!-!+4zT@L{A!-}i_D$8_|(=%n}ykiCaGZwP8q%D*iK8uJk98si%<>qa*}!n`%`Oqu)UK9F&+g9F)Yj30|+<?kq70P8>KROO8Q5e7)m@qRrCObAKTkYS=w^U*e}<{rzF4REX5RsJS~cpya4EivLE8L+Bw~=Z9lcFX6&ynHp3LXofUbtVJ{+A*_XDdGI%q|K)U5rmC3}&L%Ay%*-$B8dl63N{to6TQI``;oZAm3J3@hJ-U`}IcL3eJsol2)?Q$oHkYZ5T<V_Q2f>K9b@z-cw^sO$Sn;P4~#}{v(*_)YdF2ST^t*2<4+@W-(G&E0Sh@RRLpQ{rkf^u<zxj1b)gqgj7-o1PC{B7~;+c*DJC}U4-nLv5=#r<m^h#aL5-L*Slr_LhSwmIE;E0&xHUHt$$i{9gfh1b~42O}o&;3mMD@xxXxWxhlQT<&CUB&5g*X8Mi+5q=Iw?QG%ha&g$i_)Dm1HzFd>M^QK$7_<kFyudlRW(zZVDz{;%6~Yt~Bt@l|t1KU~aQsl+n)&eYqr;=8tQC_)PI9s0!~(9VD$wC_3+f|GN%Z9K=pjdX9HrRl?C_ygMBi52Re;(B9_zyE`Ywx24`f3#p5Z%<7yR)9H#HLAe*K^ivQ>DR`W#ioYc9Mm(IzL^(flU*YyB63q!fMV{AZyY6djol!oF`ZTwY`G5s=uQT=YFd$Il{OjdE4Mt6!cg=-ej@10$Hvv@nCZv2#N(inOy0ruC!1FuaFdZch(P@BP;0x<hLUa*BDd$C?FTPFq$Wxk^>Q6Rp%yGbZh=C8J_#ZF9(_JJzE(P|TG&$)BHpz<(3nixgW^=(ju|#?Y6ao~7!{xE<%D&y@4*SGOJ6rt<*NH<8DC&4B>-Fu<7+>L|OE#a}t_LvOyAjsqWzIwm6`6We8lzF8C4>ISq@><PyQ?YhF=)*#TL;UL*dD|m~tLaQ57-R)Kv@M9|$SloT9Ba>J=@m|XM`S0eGBP{t&0AOv_sgUUPZ*Sg+A^KUQ<Zn;hWy#xC2ShFn^7s3%6XV7}M>t((p$;*bL{__e*M8K7lDtL<kOK3jG(R)lsi*RdPi8omsN6e}Sp|IsLNOZHUhy=drps{=3!2ah&*A=EM^wDZL?;|Kig661Lm+|JkNeG_3g<GmSe{F-<Dm!}6M%&D?c!}eHt_DKIaa0({mWG_`RUpCtrCy!4B5xe?MAE}aotvsGWGaOqEB|O8ueizAP+{&F)>qTZ1P1;6g=gmJ{SO1eyY9{Dz|$zOB_V*194&hSUI-uP|2G07Z5D-P(n@hF`lBFdw+V89-O5!xlB?&JH%&rqojUv<R>#7&0khs4Rp%b+(SB~asw%@)1B1ojxP>qq_+h8?g@qL7&O}ob|h4Eb0OM=f_V>JxT?>X#nZ22XqxZwkYA&7$2)OgkSf+84dS0=0sftMWXHHC53n?|5EHN<2G%6qxq!Sg*K9d-x0PJvV{W!GS482Wak~&&`&tfCu?SXxm^Vf?CAxe!RY7CiVgHs4_kxVj(Gx9v{fq6pSB0eReH->w8#@>vA9N-gn%v?eSCi~!fabQ7J8o%mX8~CewKlq@Dh=Jj<g`f~6Xj?2NbJ7Z$-bwDiYm}I6V30{BTDq5TK0Sw;Z?J9K&FFoCzD=@`bBUr(|ZZe{+YxAtT-kozcyrw*9QA>Cjo1#o?4zo1IZ9v2-0nU5{ew!&yM%ACwMtq01^k$9M5m|gKv*-efvb-PVU%4@fy<I7x@GPin-AZtP6th=a2f$4kA~t^Pm7{2!a?|$jL9<^p>qvmnS1);&VxT7GZqB$SoT-8aTCoodA2sn#aH*J0Zbt$bPWi_+p>YOX54Xb=D5-%-#7Mg?--EH5HbT$9@0|y3vgp?^z%Y0`+;mQ<FXG#*$}$#<7G!MLG@WvuWt)ICVQ<v2#m_Y{;BG-<=-Nh>59amei3npxC6Iem+u%meQ)gc3<CVZBrl1%vP<|!<LO5i8ey`=REl!!gSg8y3oIZ@;_|bF_u<y7@6?EBAGEpDB@U`(@U(m%j{~@bOoC2swZMtH8B|vZvMOZfC2pISE5vg1liXpR2;^->@fHjI)uIY#CNXi+mzv5hvskkmObw-HW;Y!j-G?0-A}uEO)I4+X6>{nQuYzhi9^h`W9q&yKfg5Pf2x-2*Q)5mZY?WkGwcpEl*D^`Xx0b40_g+&2C1Yjq#sGVzi3z0zTAksFGQw(7n{%L?D{D_9mbQ}+W|D%%`BjYdHlBeEG9`4X7nstZTf+o;Y;)^Y&nQ1z8yXjdaV0Tw0?4MM=j;NqmJQvQ`L_Tq>C@?=*9}8z(8`jSjY{j>ZqQVRp*XYku>A9L8{2(M}gK)pMu)P?SE8|={^@Z<|lWv+%-608++8xKod^{{s+)U$uz7)LK)fHNS#`x<F%ZP)yWFDqg8Tn?%_$MK&y{$uPZS^f4m|l*bLM{D!skp#d@>HZjC+T046we0!TyBw_WalgKTXNT{WO~$>v5(5+gzY5sd4<(Z$_-I8!AnLey)s7sbt?9Ub|v1g~oQoOFPwMtb3K7t5#BY?Dtv<eVa(4s9}1lsWZ1ZuHyAKj{Q@x!2fIyOot~%3Pw$GCKSARH7`pPKTmC85%F{^GfFqirr2xu+wO?{qgAgBR1bU##>5~#c8Jpo6rBrWBfb&{1dJq)rlDeCr$@*QxEq40;7q3$e{x^&IxWV@kGYg+#v@0DiF$Z@WU?0*$?^Q&i?)N-EZnnFjZW47pL(cJv{TyPGK`<$H8#K@m^!7W<)N3AOuiGM&;_hAe-OFS%o(lzHNussNk{N$_gZd4ScnB9pRG<1m8LSOyY0;>zma#v*MedzWL>w4>~9CX2==ixHMq?M-Fc3WX3mPgQ8bbw0?m>r^g2(e-+o%_)T%$_UKJq*Ruoz!tNR|C-Nqdh6WPRxj%YF5rtMXr9pxna`h|kaB3EzJXR*}ob0H~1=O~OGY!0e2V<=dR0wD*>~~=O6}?oh!WWCS#Q<t@0giWBx(kkOZ=+f)>D*X258W;vOt2*0O-z?gmq+fJ9`U8=fYgGc81Onr<+z<+<S!_^BXME_n9mpnmHP|}`9D9jQZ4tx3?~i#z}lh+5A}q<!c&o5F<P00PIcoUUNt$%+BGm0yo}S2v`x=9^Ld5d#r%-&;e_(Gd>m>k|5#OtJjLrGQ~3?A=(njp&hK((IIk6GaW$>hL-ZG*t5@_DISBNS_=^Yje$w{q=Rd!B^%9VP5J*d<Zxen{&~8b2efe=#kQof7-qTjNA%;eGdOG3Fn4zyip*oX0TVWs=3^R^#wqY8<{lmv!-@SYF66lz^0b=?~GypGqCBvJBDJlY}GdGj7+m>-#q=sI8md7!e6emm&4uLWUukqdtMjz;FMS<MBD3s2&k~A~_7f?$B1QY-O00;p1xLaM^<@0SRAOHZhU;qFf0001VV{Tt&a&LNNZeMV3b98WWZ)0V1b6;h4VQemNdF?#elH16U?~2erl;N0YPy!*6eM!`CVl|qUrLeA0_jqGXGhh%XkeDU`430k3?tkCRtP>#E((F27UzYZ$fkIVQR#x71{P|DMnyP%ZOtWV^yJE#{z0R}4$B!Jxc~umfTUM_*D@uNq=1m2!Ntu7;8M}D>kyYs`3pcFDtGX!jh*wpbtvqayu`0?-&aT2u8i#e7Xa3_y=XIFH;U>>Gd;R&>VV(iBDk{^W_Swgr)n%Aw1Rz@T=(5PutY#J8@Ce%@*pgT4d=s<kn)AYY{HO?{%W%b6npJhU*@U8AUb5>lt$D@joR9|u!iv}UAurQaia7I|f|u!*XEh*p0hRO@F{q&#5KEejP^ef4Ln?U<c=DJ*={n3-JobrLj~@YLwk!yerfy0e1T5Vad0DeC%kr9Vs~$gk{79Box9|r4_;H@q0U;bLZ)+aqF=sP&Q8rxH&7o=n*auf(nTE>^j+nX5rw^ZlkH3BhF5Z4Qd-cn?<1y#c=d+JLzkhY!E&uS>_pPGA^zkDGzr6Bc($f;=2PhQtBy2XdtNVeL6KA0D2+LIwmX$#)s;>0&I;_^4bgAEetMW|0Y{Po3UuvMcc`3t)FJU50$!)<a{q{92l61o*bzv$wHdnM&?;ZjD3alk77SJJJqw0O8{td3j#jSqHnr(5*!ir^uY+OYtl&Fa!e!#|mJNt6}?$bw@9cS#1V8Xw=x;TTk<B>ml{OI$SPjBA6KMVdM3Mc+}!iUGe^{+pDIXj0E7pjepy#ag5>ZaImH)YS*cr;=-wJA<%iR|Q#JuDpn**|_bS1qA%<R9Wc$N10b01<lp2xg4M>55miiz8Okx`sw_KKyDOPL5C9f%tNr)@!1MsPE?m&s?WmIs>FYBHQB+v085O=n^pl?kwF+xLwBKlqG)2!`K~<CP(ZE!#@L$Egj8N>bLfr0w>5NFcJX}pQg;<+nV1<%&oB?(*#v`#R)wR^+&@MT&xh~*%dEq_r~rlO;}oy638O%-gq#R{)%TH;87PR+5L?VBibsW?1KT?sw~6Xy)f#0lM4Lxr!cKJy8sS8E6coek(U8iKM1mL3pznowI)DFe$8)c7YFahz>Sql<Mx2^F=(jS#ABe|@@oK+&EABY3O-3$ok@ZqS2_R4K-PH%Dg@Nl%xUUm_}sCm3*u@Mw*+NV#+ApClyBln4YQv`u6sZOk@)onl;O<T<)3TdLFmx`C5`I;EQx^<*%i{Fnu%Tk>J8|CDkK#<b3od_a5xrCuLX_Pz;gg&_d}UqSFRk3HCUL-G*&|ewFQJ!Ba{<pjS(zMz5&7v{Im*6j;7=aW+iuJ9cpcAS>5DGl16E`3ATCMfKZotUISfVgsL~}0ce3AK;WIKj2!$ME~Ck4a*EH1bB7J?{o>XUZGaoWXi!tc2sNN<v!UR#H&FaBuixZN77Kn$oG<+MCN04{<>e~O(yzQ^*czCdZBw#mQ+9`N?-h*&;(#G3UjGBVA`1hwppHNuNdpFYqypJ?bKr372CUwURdp%IDk`9n6P2pDt}Mi10kuVusLF46<VN|n!12=HQPYrJtXxns8ZsCrtn0vo1{njRX$g~4EIxWI9ckuH&S#Kj35OX5xi6bckvUrDDR=;Y;k+PaSdi6)8X!4MhEu59Il8DKAE%6M1~wX|z{!kL;9~F!V)*aY1x>Zdm#*_9NJ^v$cv?6R^cMWA`05M>3(XQ65JZXAfc<I4#ycGGH`E(~6`bJ^aoRRj%~Dh&X>!ZZ_X3*m%|XR9_At5)h!%6v4%L+s7N0pgV=zDG3_Pf)21eV1`AhP>a56!(8JNId-(Q>u7oURjx39jO`C!*soCAH4&q0ZyQ-v+IzSEvUIpFzdVJ2>q16QP3(bSa`I57m^kYbb8w&dXi-hqJ8hx_0ToJyd11=`=IBKIIT7|e&`#b7#`#P`mW6`fIAbFW0W2DaQ*D7vCQIftS=c;c^*ox#1t!B7P;0iy!60=C}fSG;OKq;6bi-_-Bo(!ofK`db`yzoSzs^o7CUBY_aSL~o|$uClRMD{)@Kv~GCKnWQVsAXp*v#Y?GweD*=88=C^4dMfa#c<h?ahuxkB_b(shyu@iTu9$B>a0}wl^fu6Xi--m0XA4>~0p)<RbMLXa6gQ+X7!ahggem}7bdieP>S+!w7M6iROA3Ajva2*i!j2bdbh+Wel#}5M%V-T2r{id3oX0RmZ~&bey?79l1Vxz?juQu<Y|2_Z@@z-t$oa;x3qWYxs%D7=vOgbncGdN7_RP7OPa+&6&j^)B$C{|0{0>CDoFWat!efc*d49P7t&A3D1|JikP<?VVAgYtYDKNe_Ff#y!`cJ@}13?yT?=53LvI+7vYSkNbdkc<??PlO|Low1KLj{~blv$Hk29x26K5wv30oo_u3(U=q+qX-ScNqn^Es}K2t2Gr!hRr70<Q2EaLB3`vod0Y9N(JKzU=?sLXJbA*0po&?qwhdSD`nKEY1Tl<&cFr8ha$X%P=&qyO0g2~4mg@9GikhoE1&~J)Hqsvw*V@S2JBy~|Km?=JQ%1+kYHrROW9OUvD?GJe}Lu?pfAH^3fkwErkAhCX=6+DhQjg|Y`$>jd&WqDi9w+MMtK@Sb^()8xU%c%jx?=duxaTmEZ%~M4BUakljnlP3L#;{*RHNkuQbUKsm&W=cMzrT6g)ktU|lxs^}8=@e(2q0_a5}SknASV-;u%|##<B5&8INd77u=t-S%MjCA=n_(4jG^ka!G3g9_#$0Y{@}7K8>S#6h?PGUh8x9_nN>I3G)F|6gftKP(_biCyg4l{PqyXVAuzeav8z3~m9!oEcy80b7GVtD%UcXSO%#F_`2wM1k%%;bJ@TY-czy7#%p;U%7y{Me+ri`J@U4ZRbcCG3wUXGUF-nD{$ppA_d7B266a|*||<Q)JMKrNa&0>)}`kGmI4r222TSCPt_vWSS#Fw<#zTFoKDN>29aFnWhHRJ)SUL#V7BqStx}}#lEvr*dW_-=ibue9WpPmO@Un6XkF9ySTGum?W6`od8;>->DhQX+o3{)+WnCofX^Ztrthx<vFlpmSv`}o)1slp&rM<Aw;z}S0Hcd0vv;z0C1LUa&f_kq50;Cq9{3H<B$v)sf6wmLO-vR-)Sw*xn^H4PfaCQa6EAt|+z!kAltQ1y3SfA#NPU~FqqD#NFgLGHS9nC*r478i7d$_lC%Bswpq7z=}niMUy!Od{yZTE@>j4lS5PjoKk(4<z=VxUq_ojK}hM>Fw2tGvW0Nj6*Ipxt2$1aW>32#;L?lg%(VTyEOAjWx#R1lULE?;0KGjXqNJV}HTPP&`1rz=VT&DQ?}tV25974Q4!D>==kGWt?Z=UIHue(&<tAt2~e8z?YRP^gt!p7)dFlLY88G_(~t>1znoSfc-Qx{MGJ)gB#q_D~1%Cun--CET-FL>yF5ff_TxJ?A>eiHni(&T)h^8T`J+P$<T1kc86pJss`5?tHg9K;nYZ;*_csx9s`p*2I~B{k2s3!u&+77TeQSl${z9quo=Idi7YL<nKJjLB|sj#v82P!VOg$JP&S#=%~q+U8BBTv{Z%PNXyVb`D}y=Ak88VT_N;@)U6E^qj#5@X2f-LA0jr9TB4k^=gdSU>7=29(m!gEQ&hiZXSyy*I1CNtfV$av%c+g1|i=HU|f-{M4sG#R#a8Lo-sKbe^3NwcJgQcxXc6patDB+xAgK*%<hKsgRo?H4m)SmxRKM8iR2|5p_%N_2ui6w>y<6Oe%LZtuLPM!Rgr{Ko`S7fVt?aHMwTykr&%h_af^juNPalrA!)D30thg%+!dZuJzRdX=8<VBOa4ZbP5fY*6>i4o$KZ}aljr$sRYl}XDEJqh0#NNI8sNcF>soo^7d!8vUpLIW8Y62l~N)Zj%gX$5X(*J)Kjf|MbI0cKg{aL}^wT<R6RE$UlrzY~nMmqJ2ofd*XU42X+=iEdp_c_|$LRVf7ZAu6`(-AJ7(Lt{yzIU(*5caBe{OK6NhrhNhPK5u^~%KaW+-T_Z-DB8hC;nicxzd^IWck&D1JtV0d0(fjuSn~{5dr(wS3c*l}MQkJlG-|dLsJ$B)*-ctOd@;~ZyH?Lc#5nfx2@rw!xl>id7L>%Cq257^<|$DO^|DpaejnodQ~Mnn3}`*BBg?=jByy5l(hSGoq`!ITZ^NHb$Foqdp9gJTn@<s;nRtfB&R4-GQ(c4JPpw+3GL2hQ8BySQY?E#aVT#b9#wkJlLB%e?C5P5yzYQnQa!M=;Z%-%3V^B=Q=D!KI1>+Y3c?oQZv77*%B0G>m);SB0ns^EYtEpwHmDqYBCAK1zaH{z5a3rlZ7VF;BCy(U~E_8|Jo$RnIwR6MBi2>@l%2kR^TP;xgxFJCqF576Rx|qc9k7YN?`kIDw+x=5*=H~mN>+Q*DBsHp(mXHt3XK4(TbT9LKBRdbH2+TUQ*@;>Rv9@7D*_XY5%hHi}1xwi_G)HOIj;Pj>Ptu`clMbfoQ{))EO~INE&3e)=QQhj{LBFkFFx=c$X%(Q=a%s+VAXC*j7#Xu3L*6Nhxd_jmQU#D%a4sc<d4hXRh`<%cM59DP!j%w)AT}O`5)j{%7&I!y7ACdw?;mLX)M~!riD@mJ1FAh2xZtt^>K-!30z32<m{c;$Ruk%;+}C5QFR=NMlAcsyAlhcKJ5QJyF;nPZNO4xqiDU@eU;qYoIShq&GqU{~3rMI2IGl_|653LgJ)Inn+H!|2TMKD53@#3~Ip#^s>U_&f1n&^}#Z&pk8w^J<jdnAiQiM#_V5|ftNlS`c`h7byN+b}~MJM{O*N-sbcB5E+6Dr_<7Bxp~JSHkh6ev-OEGKYzDq4YVMpNNV*b{QoB*d6+>d^9sZv@zd<@74E-7$Q)Rbh3z-@+hZ`Df6TwjFION2y&%8iN>{=tqiQ1Y@b#IptEmfEmoRI>8wY0`vodzy)2L==_r0%EVi)6jbo#cY0QpRPxR=>oij0j?YmTt+`x+xx4RFtT^}=6tPo|O1Q9~ac7jADG7Pp%7yPVIyDNg@3@#L2SHx;x*H9EVMRGX&sihY_2>91L?0GO=@-0A0_haFJ1!B5bK)>Imk!iIq&=q$@x|S3<bcz<iSnFD=~gh$$)>5+GZc{ySc8~KmbQ#s+oe}&=dNzIDLHpE4IS`<dPfFqPUE)eneY(zwpA^HdZY|&vO*Y2dYYmdqmd-V@vEXXO?&_L8uNV)Ei7N?yJn-~S}UT=r-0FP`Au*7`#tHNjChY5jmu=j2X`U-w``@sfiKYSvqk3t#}5>_lI?b`oYpdPu?5noT_I0qS(9KQ>xe2)gEp_JPZT%piJ6ElFsd}%;65JQD>*~t{Xm9=6{x&&A$-9u^AUpZRW+dkS-%am(^3pcwA?W>P~535N3Ogak&#A&4m#B(_(R#<;K`H8V8;QnTnOnnLnpv4Xb#ctBl#D5;ugz)u-gu??dh>-JKfi~Jt=6BtrFA?_CTnop+I(L9MJ|sb#fX7rDdKMT&(mG=1B2~;+gR?%Z3Sl#00%`O@y<H=qS58=&+scXPf7se&kFGAPWVaY=D3M;nSy!x50TU6Dv)h4MUFLyMdq^26g~_Ik7@Oz-r*(Dy4NBQni)9%`$_ky<Qt2uQ+^)@MOj&d`e(3!ZFiflc24o1d~#py;Vc&U1O}iL-88?IvIQ{6n+;%Q#M-V&Rcsoy2~E;`(LN-@z+^HHEzH+7E(1$v~d<@dAAm-#rB%3&Kuc3vp?r|dj5T9A>sGFg@nG82>yOYzeT3f`)f$AxgEQ7Y_#B>68X-x7bTvMunl%T)7pdnnC<Bq!-%2G>2>pz_HRcZ2s>lFZM(f63iL*?140AWn!w?EM#)op9n!a-deWvlWVMLki8ry{BH&8_C!o*T#?Ah<7r%ZVe20vn-2T6W{5vJ&|D6ASq^Op*u<d2*`DB57Y46H^fJ57SRqU>$?O8;nHJpHAuqz;?yTf6<%g%+ptIVvv{(;pS`9@TuXSwv)UF(qDlf2R+Eo5}(y2IMElfzy}{jt?pcH6rw-Krt|n4T{lGn2l%Txrmh>#4hN<QIaYhW_B1%!3oWXL50eaf3!5w<D}v@yFA9E+85R<YopHgqDi@DV<b+>L8ptGbo7d$Tm)|4zkw5^#XmXBL!9z1lbnA4%BM7q!;kLs;1Z8oP}h-4sp6&>unjdzm$ebXC*4RQ%>o;o(03*%^O}H9@d6t^;J->#ZwwDNm*a|I--2A17Zt4-5v;CA>9W;<hle&`{3va5hY%4Lr<@x?{<Ku=;1+<vD$|pg)aea$ynfaehg4?Z_OV<)q?Rb^DCy;+&j~ZYpn*)M*ee;8D5;xNhL8Fby5Y#B%fO{Ni(J@X}oB=kjgs}NZLQtvsKAL%{<z;S-V**49#QD1}%xX_#;tYQP3c;rGIT~^-~LF^Jt$o2s2zAg(<;vWqL$3`&a^EAZi3sj$1>Pp==3LI1X?z*1>%xLsLx3I^dXt2t@c9giECy9{6svlE^S`>C{P<ba${>oh3MHVt-c^K==}@^QJU??fLvbR|9)v(smn`pjh&27dOj$l0Fc)ZPHb`+;EX2lLXx2!2v_V<!w-}c^x`nB}V>Xhl<2FTVAj8cqU-$1Ezt7t{!mGflA4Ewr7H@x#;1gCr|DYtZ;XUrwV%TyCeY8my}y#L4I&MEMiW7j>Vr-XK>%s1;}`mR*3xQaE64KJGux}5j>;u&>zKnFSd1Is?`NHPOOqdlvw>F{e$kEdDWz!D|(>x^Qr{1kCA)efUVyLrSB7tk*2&x(H8aeJ1I=fqu~`7ToAi?BM*Gpdra|M8*U^I>%0WZ*p#6xU6Iq<IU5EbtYct7)hy|(R-XA$btN}ZbPn&ex^gZ54fv%t`>KVL1rLH{mB&LTtSLvNvGnCx?bZPj*_p=Q7$q?dl;|BBM3VNA&f<y!I!9}FC7-w}pm++O-JZ;W&<>6CX*2=e$BK?p6<yPYW!82lacmqRs(Ao%t$3|x!s$D%B<F#mI(aJD|9e_cY$w_SnemvU{`l^5@XOho_YivgLV@1b&hc714uFo)&noG3*iPD29vu_s0EL6@i_=4tT6Wa%9by>#yQNvj1$F1XC-C_y&Q9XaS#cz(O)@bad!gupt4Z!z9s*~f%28oLcROoC`XjwT4K`~vFi-_b0%YstZx@zCUexJS0f^uFObab4yF=?5RDj?2_SkCE4^8~<&6iJqKl>P*pS?eOeIbsE-yK}>62<vo$_~c<=)hwK!U$mj9<i*OjK_g+JK}&+jo>_BJ>iEzy}JYB<zO@EjbR<_=s-D`4Di4XjF&@YSR|mst5}3Rz+TK7c{cDW{jme>*y#4gsevLC?|Ud*^9^b8uv!yZcL!<>1941}=qNrJABHDKhn&BNPbR1F@$=&-IvvH!I7v=MN8@O6_+rFgjG}Ndnv7l~N2g&jJ{%n#y*Pf3gHyT!;l;`1@bu`#$@3RsGzpK!&rin5(P%U}ez6=s506ie!#F&gMB$4u5074)9v;Ro;>q|$gny6XFg!q%#g!9sFF!&(IyvDSfGy*r6CO{VN6Sf=oP-IV93LjfN6%0Ah@T|O=;(Bu@a5_FbP^p#{OI(>i^(Yuqa(x;#!MKVlRJUOZ!_!i3$_WD;P3eCP09IJu9g|rX$*l9g9)u)f@37#hxjh<&j8s=!9Mv1_ey!8RdPW(uUdXFUCU6?DGG5#ga4{rHJE1pOkdoXi(%s#>mZ2pNS~AyP5m&AgHX1SXGhV|#}%^A^0c@(*f$dlgA+e<UTyGz?ydBqVA!B)qcyl$;Aw!OzjeORNuJm3UevL6I(O|nTjyK;OrDf}Cilmw_n^6626Smia5dJJZBI|2UfUfB{6F$*gDfh4+}EKm3_(4sZY8m94Ee^O$d=LVCeWig)=mpofNi6<f8!n^S|TSP^asIIG`UI})7e%$#!E9>`B>lsCV*Lk(`ubb_hgMG+fzZCZgj!1nmUX89p}pNT(nFdXfmMF4N$~sVfjbYSpowG(zUVbgI%RB=6V-lwyZ|t7(X;K?gMR|k??xY!nYgJB?@@pvT{u`Iz>)*EbQEl&|dU)Tj*Ml!FO)5!0`3LA}nrUfv?|M7`6{&L)-&m(z#hj3sEnX^v>3e-Vfe$cwJ7CaXs}T>1gzDdM_++l$zD-Db={yz3~kIs+s~IOoH08p}x*q&@_4wbhYp_%qqbEiMxh%?zD_gJF&7u{7?Ei1kp;JOJj873(kbR*r<J*5l$@WX5eq|l4@*t%~3OpyMNSZNx>_g@0=44Xp?tk`SMZ-O`|__$%ARuIm+BQ1xEWs_8pXWfZhsu?R2BB0vd2idx*ptTUw0KnZcXgtf1Vi6D8=^?aq<#UsM>bHptAS!@pp6w0(;p4(?f`&XrrXQ&RCIGnvKN9dtLh{GJ6E_STv4L}shl#}DVSw&fCGb+LoNd^Qfx;=XS)4sOXq1xuI*sNU_P;507h_1m-8e`T*fefa$T?Ba}WwE~-h)MpV4g8YI~Uz=r^3>Sd&5)Hk*F+}*{;+>A`if#0C;gO%^*Pz^iHVMAE&JX|f!}f<b_~GpjAAUGjP%xUA;Za?{8=s+bGOX&d4*TL0p};A`LXT8PLx5r#+)?lM!ChYY5Ec|^?BY542ki`ut`^#C#^&0N;ihYMZj|got9PlucQQotg`q#jNc<+PYD`AUL&ox&7YxD+DpdErco^QbQq}^W2X8iTKGAduCB5V<#2aob523;ZIo$N1wTA60^xQ6H&4j<bL~|?XTuC>Ih7;tM@-6^bd4p#QV@fQ@XI`St$E#G-2l)&iG>4Re+<P}hQRpfIX-5S%7$H7<$<MMuMTtdY3S*&#9VPYRgnNTIOp6%_hK8P92(_rtfU3*ejuGvK|4vvGn-oGMowievZ&&jSz5xi$xjsNFVUVTj$wuW(8Ciyw4*Ci`3L`?^rV@M@$Qwvzq+a)BwgCU8j%F=cR`-|r@C7ew$WbBSd1qWI1(qa<#pIPhxd(L18qUNYkF}y5b#l3_1o}=YD^?CHa|p^)g4tE`t!{dAaU#`&E^xi>pko;{Ph7`_yYhlSM$JLs(K7ms*3#|b-Cm5>WLRWoDVV;@r}Xdxt%dc<MbQMH-SoTm0Rpi<e5b`Oa1r<slpsL@0utC4+7F=udRFy8Q_Y+=!W{wKf>#Apy^dll$n@F88q_EJKL&?mlNaR0O23M>e4%EwVou-bZh!kT`wMQ4LIwBpr;E4td49m7QgJJK6==4^ln!Z1RfyXLXf_$N78t*^zG$Bjrz7H>O$~aRUhjITSyD&C{>_!rq<5KA{p89V^wrubEe`s7dP8)(k=hB^T|rJ(p7hlWj%ww&4MC<??W8IBa<&v%a^s0>(}H9Zu2wQe?>`)dHgnZp?+4l$4-oAb)`L3oDmJk`u~O%mv)+9neTT^;?s#9N<<1FbxO<okFLCSa!4cHQ)UDXa+Wvw#QPSPMU<sE{Tt{$+t+^ae?v?ahhvugJk&1SbuBLyqyq$!9w`<RYTqfBs$+Zt`HAg55y?E7AFTbY6+|GiSTo2O=R<`fZC92x;0M9)&zIz=2#{6vG`h?d_+S|XD;WZ%8Qb`_^#E?e#ZWjdpFe=%<exQ~QD0orW+z*Vtkwq<5*8xlq);dlR%i<8G5ZY5(9%66Cvpb;Y{6MY`9xSH*N%GLORf$=b`w#z6wzgOAGzhUoXDaC>yzlAc;`6gN@c)mpEv@N6IP*-rCY@K&F9bV#4<6_>C!9Nn?s6;BN=-Ymmq#~X;?sPzbL2psh@sE8ZAa3}Sqx<htEk-+lQpOhoIf+2QpGEWjeVc)3D}Ln!v&B&9}MphLMPw4W>|apQ})k0a1HPOxv#Nf-BgU>hV}`V@Cn#8c*rK6W1lVcBoYTe#YSX#TE<k`&)z;?>&Tb#`!Cg&c?@4MJO5I}l=|``v(NaYKDCVre|0bIOB4Shfc|b_sA|^??SGI%mgmIO9V^=KqCVJ5wtuk?wBo`31>(c}vHW`|{~j}}K(Yl-pOs{Y3SUSD#Is^?dh73LI2_`C({KFKWjS;90eG{=79dONa|JM-yK=5paTfPXyeH!TsEML_l(fnYd&-VY*+ci;5yy(ow6-yjx=O@+X=#o>(2b=vS~|mM3?TfVM_iwxykhFaQ8Ou$WK(uG9*v%mHkdy3Cx8kF2yzi#7)SzG?Pr`f4NSetIckR>&7w^cbKHa{%Rr*)cloan@@OHT^?l*e(B1<k_8wsG4EG-Nb3J*{cK!*Xd&~KTFpxysRv)w=+g3w5`k;F-v2T!d+c76+VvoGPwHAV;E+I(+#F`A)plaxb#q*s+62uu56Bm%v6>u;ul)c28!-jNm_6CpXN4H{SiEf1rV=k!Qkq4pl<w`iNC$}yu#><9jzpsn01Fz@WVH;T64RB<h&&2QY@~N<Hs+pq_XpVP{94rq{%zE+~Lm5xomWSWqkz@9ASn)G@gkGk7Lwob;-TSj&{Q9PD4^3Z5?u!^a0Eq6tcZ8zr|JnHPNA9+@nE-Y9rkgymGBd5jjFprzshCdOW^IvZXKqb$^9v7S5N#aof#H&a`+sETczYERu=BaKo#LSDc+maqxV1ps4NQAwg8<h%gTS#idto-~zX4E70|XQR000O8__$kLPnr5gqYnT8I4S@D3jhEBV|HI=Z){|9E^v9JT5EILMwb1~U(tnp5HNv2(OP9v^l)ZnN#0~P8+&b4YHGPG3`jIdj6eYC21zqq{`WoicH>Ff&dj1pBG8Z9-S=_s18^K?@KYYeqFhUH|K{P<&8t5Oy`BGEF3Rh*Oylc(TZ)HY#4=A~Eut(IdXXzB3MJ!YQ6_n&MV`_0jc~P;;=|MZ<9|K{o7f8m1A!~q6uBzJViz(EZlgOfsoR3L`0qORj%Rz7&`B(kj0Xm#ijpi$Vm;<=x;|kx_bD00E(Uv|e|;mCX&#lguf+6Ls3?w<_QkU(la17+EVOX%Zp072dA_*)?)zCy`QVFvMLdn>GSz3~zY){tnHXVz=#e54%P37>bK?y$W0^*)m5jL=Yl4InBVHf~0{*QJzE?>p!xl4LhaT>Oh2A;7Q2Aba^{VtV2A4}5dINs=q6~EzsgiDFhG&+IXZT(m$~87qFI60{;bktS*Vih~OY!#IPmdpG{3ufyVUO?cAO7S1-=1cln{Fh3KdE;p-the!zJK8R2flyg`$xXV0;EHWUGR6EXS2cJXDOjd*>(exq)_X-;t+^uD4<m`EJ9)xOO<bg-K`6qfPRJ#zl6N;*CLk7Xq%SfT6LkgQthM~Eb`*u1>z&Nn`A4o)l&1IT&<!k`AsT)CFCwjw^0eIyb5<Om&qcDQc*;UmuMx0+pxy2{^9ATr~8jjo<G=Ti*=Mi%D&iUu~bZ<gk}k~O?hgv%D}E`s|?uBkC}MSO!cvX)`HG8VGh_;CZtg%P96*#*rhd`NUaL!uB@KH)9RKOTHWWmx`RrTi7t~x({j)QK4uY>>ol2HHu)ZpgK7|op}M!ifs~5DfcYJMc=zrZc0p6zTn94SB`VK?l`LVnEx|x*o#8s)$YC^J5S$F-sKnvQ>N(u*=V7H*j^_<5Q@=#q4*7kw4P5L-M4Pt+$^<T${D%HL(7#7u&4>Hvr|^gWM)%##U@(9}2~6IW>w`<$Aw6;ZXED$7)ZmVi<yK{2ud>|%Zj-|Gf;8Vt<$45IM`uLjK}OQ=L2Bnz(I%x1AwHMHFy_<p5-FrLB><S2h4x9gW>E!sAv0HIi#&!RPn>PJy#5|!Gr3QbOimofBOxstT(o;6iY`_P9zpd3mc{m#V8>vvNJ;ewZ_Okq+PlnH9{-cd7ik3WC7Vr?7kSY%#)EoAgGPGDTpwC&`s-hV^&L97k!;Y85_DVux#<XRZv5(Q<k#TXrV2*)X$n#Mr9M@RW}5tanPMn+GAlJBQ*#kg3xoDJD`Z}-{pQf{N1LcTqqN9jEIJ|BZpab07HLv29jv7ec6>dZJquaxnaAoJ&G=xr4^43CGvVsIEz(K3ScAbv9v``%S)pVhqf*RextB7t$m*I|EbKKDC#vP(jm3+Uh4JGB;EZ=!X^5fDIvhftWf;QMr%Nbew2@=lmd|KrbQ<>5K!(@oCQS)w1_sbix5Fr31p-9P6T|MfWGOPZg4XaL(P5mx^qN)J3Zev<{WeME&w2Sa2S7h5mBR#_EN`p2IJRh=Z1h3bMwSt_(%#rmIJM-Q?$+&S>H!o)v39A2LzA$SpGw#BXz$!8KfyFLu!5t4<Ecx@YA1;OoM*Ddc0vPfnXdBGFrC4JaAV36AjZwkpy0T4jx^&G9+Rwe3xJl~1iDC*(!C0=yy<A>jc;DXCub~*4i}BGDp3%wX}AR$yKavqRwEtU(;{B5O7ppEo5MfAVu8XZbC7>66!1)yhx%A`V8>(|EX4HlgD|EUE*Wk(Je^SfGvvvbU`zk}0>F`)JQ1X3D+EM-jSPtMXXM(HFxs-*2_0{GGxMfsnl+Re=r2&_*oYLlArE_euCd@siyt1}-ic*GK?j>~QZh9b-6qNoa3@)u?>VH>-fQu<MWX&D5{<AJ@vw=0t57*0WAB}#-h$zc;Q=CIcsJ#=n3KwEb(uZTyJq|B+?bL%GsV_`qIJ)L&Dx}~>#2hsJF`Hqqe7CpHfx&bzq6t*qA|^b4Z#!4&HewfJz2BjJ!M6ZAtg}Jp~KmSA3qSfHc7%X0ufKR<RH)i=FSgQ=n2rgkYdcmwTKK`eZeV4<Ag|X=nxz~FpeX6Py~q-^p3v&7eP)Xm7hQj!KIiZAoRtL1k0KDFX5WxO6*3#$m_`J_i>f7CLBD{eDIpH(=jn^?u#*Y9G8bePB6+I&FjN2x)Qx4Hz3_lUI4~ihTmW4j1}%4Yw$k}kJ#O53|iT>A!FI`nv_z@)rK83e0>J&6lKn?uZ7zBFD<0Dit?$^2?L=$t+=28{iw@oz*|b%kZ!;xv&zbRivQ6Lfp;@5Mn<{FGxE?BM1;gWzsdlye!f);$tcihL6sj)#IM^3pl}cfAZdw=Ts^|XT_X1;ui=zS36o2376Z=OyfUgXZ_Yb9>Htx7{0SIK=kAKB=W(V-r_L<l3)HmrIHQo(oZPWW?_xmJH1oJ04C(e#1NzRqO6PKFzL`~<M_RSTK)b0{!|bc14W}`aHe|U?P|^$BClle|-XS4WDF7z4<$;5Id9n7@C1a!q0^aG1BnYt;sZ^6GTz0puAlC6_eOs7HrX-DWwzXhs$|^<~Xp^!dWMV_tEmZ}O9>Q9-hI^T;)}<9W0d8>9bZ(Hi!n^BEx7V??_flO~P%EHG$oDKLnq7^>L2QBF-*aun?NwYn&YkxS20u5faJyTqCiW^bd$eVE1f=(O{aTDTMBA^Wl3lEQt|G(A;{j}&@{xhq-=xa5``m(4K1VZdZ^l6m-C<{_Yc?b%><bmevTlE`;t#^6vR*T<s{$1$Vyj&cAotcGE?Z{ZChYgeuyJ2pn)xt=M}bD!!dG=c1u?lY%h~wM9`@5s^vR9jL(S=Y0iBtPZ|x&d?25Lp{Zzke6u6b6{nH-dxt-Egs!5#970eQe6XGi<97PC25ykW$`J+^x&jF)vtf*hCyA*0o)jxm%z4X)ZF4(vFihYMzrBF7VV;*35b9f4%iQ$aNHd*2<h6*!R)Oh8c_{jY}w4uwX9TBHUobot-IXm{!;reeYMi`4@Y6Ie?`irIb-fwlbLVR@h3j?ps38>{rcPG!WwV3m|s}YqcxPlLjyM&zXDIMP=T4SVm`1s41$B{aGy%_78V6qb_*pcE-Jyj@7_RJYD-FtN_!fpi%4mK~8?wSWZc~)Bl8=O*x`3paKmnulk-odbcpo~Nun^2|E6sW0q)b6vb`QB~oS1U%S!vXT(3*-Dc$fSv3ggsyrv4%_A9WdxF!{ZS~n&2Yx#=-6K<g}XR=S><--=T>ThQT-D`u}~gM7h{v>Unhy*|h~+ThlWK(Fe`RQUKqBwA6)bq|%`&Z<z5O+p}6Y7CgB|YfG6`jn*zA+a<`l&2B_*H-JaP8j_}rxh7B-2h45?$kTLBo%taIK9vz7f9EE+b!KhaM=P*#JshbBAZ`^}K?1%<;YLOo$7i^|2}Z5aUE(7%z8|TXX}45zu-&%l?z5EcYD&<p!py`k&$N%r;(VRVcG=xVVIftxwv7=0DU8xc0ZzFcv89PRNsB|N6b>4j!F5y4jY*V+(Of&Ts=CYEaa)LD#PHd|m@hg?gFaj2_Qg)Bc}^WvdqxM_byu8JwynyO=Cw?|Y~Zi`a_-2PlPS~zx)>FJuLVdW^2xClWj11!?3ywmd5^coan}PNibOR_a0Epj&ejUeT?>G1VbbRyt;*o^l_1UW#m7ezyydEuxyT(3du~<6HsY;{_(Z-~aTMb7F*Pq7)>{bcl|k!x^2Ky!{^;J8+X#S&AwD4-XA%TKEv5y4osDj%wf2Q2`zbS(E0lFVb;?~}!LZ(|%O%5$<uX%TstM3`+krcXQ7JJrXEn*hXQ;Ka0pF&Ip$(E+KB2x$4AzbduwLB6&V@M(_Lsy_&r33TRa}Gk<1VZd--y}@0b9|<bL#3vBLvm9B*48_b??>e<aBs@Y=*v2rE<s1y}(v@^OeAw{Hhi~ur25mZfmE8r{dd*ST%<xU81I@23qOWloqD#);J69R1oY!!wcd=&%f0#uj`n|sWnqOq>{ZroFXM)Ev(W6YOEjU@i79CE<3200*1uXUem1o8eLCo)tjB?pdGWR>3nn4BoW(868Q=bur<pCMkKB&?536f=k2hxD!vA$?BjZs|Iyo&!j7Y3zzL_iyjs`%Ye`d%;#9;5v@2Zb9oXrd*A8p}7T5WWha0zErUumFnxmc7G|U_wcW3A_mb+>SNtokS`}k*@VRF{oC&$289b}VaZUb^J@-RuG%{-38OLNHS%dBR#H3O_yd6Jqu#d~0}oev<U4+==)<3EjVPJ%^FM6PBWP6@aHZZvM*whrr-+q0ItdZt64EM)nw(O5kF@pg2kb)9PoU(O4a1qj^6C$Tio%lbJp&zXG|Z|{d|(`21Gt^8`)evMNo%2^FtaB31D9VjXV2NvM`Xoy4L?~W>d)}PvV$Md`80*kEiYb2FhDyOu_+@38;jCLzU_N{VHtA<$*E{sdj9qTyrK|SJ2);VE}6-($Z)2tQzcMukrshTlnUFZ$7^Jq_YRU3^}0QT|SULDtctlv#{qwa(=UOOH&Lu`<>LHg(R^0SEiRrQfXaIaQdDj>h7C*{UcFH}+xK1`gie&1wQj*1|P<1n(MT*rKa0&Z}X<~?VM_8q{a9npbcF>$J3hvFgpP$+Ug6mN#&VJIGl<{PS}%Ke2kP55(p**aKk);#}y#kgeMeC*=c)r=RhYvO@<r!=vvoH&b(QQ2R&iRy$97|AotaMz*liJWQ+R6S_|y=tKa70Y_l3`|w?jCEd$U27UW*3b}2U(uo`&bKus62+t)#vxEa-W#lf_D4&fqspHtb(w9*au#Y%wnZRp{S*@BfB*RI=SqCois2|9lE}iuDj24x09<9i&P*GhnJU>S>r&S)&jU{;){Q>tqTEF{&`w{=f$<v?Ggg&KuL(}5VuB#D`UYkwZoj*^`2u7c4D1G}Oxu<gz4N#-;G9WQF~Yt}on62j%GW{X^go&XUPK)EEN=JPKaas(7^xUlX^(3cg6fgZ1*fxgx;cy~CYf~RB-|tMj9zm|fR=&(t6A&Bb{s1$oH4rscnC^<YJ{z<jJ$jMwXltEOr}UmN^j%MTj2>5Z{#(v+gZq@vR+Hz)onX1YHJwI)GO}VR{NPDe~Z9hDa6}Np<S~hD(uD(;p7#6Ae2y5kTfj}SzW`BKsOAn${GY7{2x$D0|XQR000O8__$kLD`p2&qZt4IIynFU6aWAKb98cLVQpVyXk}w-b9HTVE@N|ct()79A2)J`?*rsbgk7GBEM6wr_b`g5tO2jJupB@Ce$}l2<2alv3(&5{8r8qMSicj=IbVNTU(Sy|?JwW{Zb|CD|J#3j`S$BC+y3$GufP6cKmPpJ_p2X5_{-nrOH8s$DJhRVL&_=jSaM9O$5Ty1+-J_Vce9wfOv!7_(n3mWhNV{YUOsyHv1*%BQdO&(qosAk&R$to8gYk{b_%D(Jwi*N)YwXBeS~zztZ~S_eDd-ktip3<+qq4XIp=ON=02yd6=g*0Rq7tO_Zq?pbuSGy^6kk|%==zGd-<Fs^1w-Z*OHBAY;DIgdA=O@m3OYk>NtE)+cWa_CrDi<Eo&0Py?pWVJep~q_3Uky!|7?u)Lz?)M|emZr_z0kEwUc9^igv=afGr@rg``B;^j}(8ua9(wAVsNB`@AN<dn0q;#o$~VZ~M}dq}Y<w!|cA+1}85`Re8S>EYPeA<VSu456H1T^Wv+&*D+L^iK2V$+Z_@S@Vh7a*!5kjh{?T^YXTMrrxu}q$B4oE3S}r6yBkSG@I7S;xfK%#*}xrLj6|O>Do)2_wwrHna9S{D^DqP)XIvkFb|X8hHlBJ*F_UzF^jRPuXf4f6dwPyI`h4Jf0lO1Vjx-Hx{bcr=u2gcsd-cQBr2~I53LF_)Xai+Nl~paZ08fy#s{^@LD^;Pb=v7UjaF>;9j&m`F^19N%o=&v95h=<fo5*Ye9+d|)4lwI+MEwB*}`Jd!cKMZlp~cem(IDGhTO+U+47LNYmCx$F9XU_#u?#0nDIevvY2xlaRw_hX;2BFZ{{g?i_EuKo7sA@aG3c$wZI*nmYtJQZtGtDLG5VFcF~mLNP1*X>!Hj!jxGw;G*0iS_X+LHFo&EPyl+z%tP^9umw!+Tux)iH%;tdJc-ln23mlePVKQ?mJC&g=+Tp()TAx{z12e4-E!@k$s7*ZP(2@i+Di4={*y>4Z8?{IpQUL&iUKg-9u$=OzSuMmRG2SUHKB(Q+XIFv7RCWcfae&j^#)0`)p<%0(;{*enQo06SO_wnX{8Ksi@(*f7r?%qb*Mlj&$OF!ut>pz!wZ;miCS!S^x-c2l)c{-wF&xz4lkyL0sV8bjnw-(9?aDxRTPf6TFc*kD$ib>Ca}O19Slbv@+uDcmPw(X))God)?Ahy<J|YE~HaoLN=6gbMEA|7m4Mc(UnULC%th=+~ywSOHPI^!)RTQBZG$oEv1{7D3jy%FFJmt(M^Jk_@r{xqom~N{Amc?{VyTd9ys7<x6Tm*eM$W+R+lv?6JXwk<F)kV6|q3hk!1iCPT=DI5g+wJI`bJC02T3XX$B}nSI1w%_yz!X5~gUR7z*6u;HontxD{Wz;YHbo2Ii}&&mYA3TkQbL()mtk~XbwP>;kS?AYVuzfW%oZ^QaaT>p)>yt$yM#NXrSfu4M#bv{lw*v_^YTy;2(yM*l(<)KtU;<<4Lagk3e&U$Yn0&5PoDlkZDV3`tG%tWCv+T~(UyTqXjw;W8wR6OI0}GN>W&4YRYTEg^VHw<{x|#S_a3dU0?Kh58>aQKm{dc(W6Z|-or)V^nwnb>dI341h%;hKa9V4HTk)6cflaI0cTGE#5kt2sIqPAHO6x$=)MiP1fzd**HcN!8V-c*3&~L?`nSqo5b+D)be}=a9HAK5j++H8}LWUIQ84^Q?KZva@$1xNI(_0v~;;%a!^tq@ugw1t)wa^$L)O4a3va<$6j|B|{AIl5OjFMT;8^442--<u;66;8#(7@1}>{XYS$tup^m>$7ptX0v-%|j#x1xaKDa@oQ^?Q$#rPRp@46MhM*#6pgalEknZw3hc8MF0)W&e(=L2Ol>F^BOUjW*w_|EB-M2Li5Mrv0x^qprWz#xz>r9%5(^CiYw&tr!s^FZGmEh#gsP{Nw?w;zaJ|b4#v!n31>4n!c$_9c8@w440UD2NQOLP!!X%!IngyVH7wz+_`~i!XmIb}W^%BMj+Cy_U|LwFrg=(mfg*d<P#^a+K!n?d!n~YvEB<i1lz9wyUKr`hdIlUR6otnui^p8N3&anC1#Y8QU|57@f)5y-@)4mw?BDj!zkd76{Px=yViJ~aK(D8eJXQfPnYtAda`3D(9=;6V@nBg|Xg(IBs<Czahkk6x5A*9k{`vLWkH7x-_2;K{(^Mfj9Xhd$c{I;0y%uN5xso*SA_$S5-G?sSIg{Z+9e@;ecxwOmFJJ%XpWh#~fj?%{Anaxqnm7#_?7zz)-5_^xN!y37;z99**Qp0K$t^{}cKzpPNs?zPJqbaE;pk_=gi;hQ2Nscm#ssd7jAG0b11;jB&oW@m_9}hf%RfA|)<6mI_kjWx)vWxPB|CjO5>qK*xrUN1;)c$R5kGh*k{ZZywBO3VNzzAc@3g>!@(I|l#{g+*J1;D@WCkDC3qO)mi?t%SkAQay0JPuBzg$&V9>f=LYMX;~z%?}g93<$Nk)-3OSV_`2q%u^A9g8T~`QGJ3cq?Dt-Z$a~ZolC6d4)=*MDcc*?sE}PSo|<WJU$f30}!r|P0!6F&>ZjOAHGU-c_AvuJGPM&#=kP`XsU2X#2fMr0KJZov2$A*nUf&U`}T0M<zD_xk_5>mBUkIZyxXHG(w3WS)*ubvHXu;?i~?CAAcL7whk&$CbmU(C;VkfgBCnXh8y}QQMI=26SH;U55;C%bVTTQgY*=?*1?Msx*`7r1<@F_V2AJM6`mo4OxT>@ZW`8q60qaNu#9J;3Lu{KF1I|1YyamkuN$(%t0(p4!wCL<?3{ikQ(CmchVKP`x<PWQgq_GeTG2L15;5?24Cfs55Cdth=V`d#YPq5&rDU)dfoQ;AYd<kVvaT$pN=ZBus8p3$O!`S`l-z2#p8Z4F!Sp)#mGtgyat6PX3CR#du(_{g<)S0LqAK#?_&9IGkSiMQItD(;ef)ZG|W#A%Nf>m-NZppcG(EARH3;JCe#uR@IVb{EB`J_1yYO6_R<ZEJO3*@3lK`?>X_mz0uhMC6|$w}_rCpr$=lpx0Q#Ip5X{y{C{M$3tY)4jzRMn8zEKHlaH=!ga(ClB9B8OQ)|@=%kBu_uL3n)9HRoC+-f5vH8Hi0@_EFhv^?<?EQ-A7X-e3&fOP)=ryv2Q3kwmhR;r)DA`j-OuRQUa68Rh>-+)3FuH=q$Vi8RwWTuMyCundY%X{Amg#TxAG5aqp#m0@z;+3r6*($r6cBfFGi+w5mMV>UEvKvv!ZN-=UyTVq4Qq;^%|aspfX;62Caw@{KNXfK%pawg@NgvR-}^r>$qyjAR{0P4Fhp6|DZO@Im~=;IzjR}Xv9b;R;6PJnDV1i6UnZUOLbO<nk8an-QXeK8S_n&!7UFnNq1L+9%cH5$2ln1oF^}Li5{7@xZ6W(2sS*=5OBfh+?(?zNjmKjy)o^D_k}dck*WBeuhVhBuWqt;gN9VW((aP6J7m7Xv#&cLzey5KA(JntT(gS`re7-o1X^&EV#*?pm63w1v6jA2V;vnbBM%imY0iV%K!C)v=M4uI0$r5NVAz15gS13!l`wVlQ33ajEj74Dj+6kEhJ7E*H%YRvS?Y1dU(oMNbf@m*swy$_75^7;s!R}Vn+rN4&ZBEMt{ro^mw!-eEzZ^kkj3_ru>=i7I;>UN%fQlfRM6daaH?RUQeW<QMCwexSF1lH$pU@XzG&~Mc@#H+olxWZ+`jiPhudnBFfRr5hgAXe1QOF_yau}!f2M{IAp?Sf8(l>4&V;Q4-B(>$fht&#Nbd%1a~AWR_}al<DfB0;Tk&URlz>Q*!0PESp*#bFT_0otLNV?gzAxBA_5EKZ7|;}y=NSW7UvI_ViD{SvPXxHFrgZ?JTujio;QJS(FW<g}TdWeNy6-T9SR?YeH*0b${>)2OD=U$~;$hNZ*3b#qhJD$5M!<<NdI?#PxOw3C?tgAFWJyrywJonRYql&i1;HZ)NN=b5;fB-k!T7ehe9w%DAwb|Va|ybj9~=+f=A2vc`qBVHRr-RP=rI&qhX*~tmBJ(&L&iWjB%T|d+`c&rCjmM)aOTG!|9|m^-)G{PBJPuU^yD=H=_w=5EMvy$boZ;(H%<3heI&FH4Wnr+mG5HTia+cwAx*3d50^*R5!S(N;1rQ%9C!6kE9M75NDA}G$`sWT2qh3PN$^&@zx&G-bB>AY8KW!3H{#0<;1#>jbW}f6%fXp3fm!$;BC5WDO1N$1BZxjGNvJdW4#Fn8xO9@)pY@7xnMGpXi8Ng(fZpPzSP&ohybqI!Z~j>J|0c=A>eIVIY^1wFY$T&4CJzc5;1K_dQN#V!f+6-zN>J+i699}9@znlPljMJWmLyT&39djn<g`aHhs{3F2_WxP<DrjhFXYY1W=(QQJD~?`D*Dmx<sY6pw2w4%_=vu3iEo+CrFcfgUBrN!hi5kQs$^`%S@RdCF)PID>ZcpG^6AY3r9{YwkOAEepk}_AR7=EZukNQF>p7=P$Y5#NJd?59(+LO@e=q-V)duDrqk#Pkn%zpD3{hxxGHqmR6l2Vg`Q&h(v!B&PfY|n}wem^tAGTVPr!K@jLT)+(d$Mo?<lJ2Xcxn|r!ptx;B@U#YRr28Jtb4XqxtD+VDk1~ACSLbwUEXDEdx}hP0y%yjzzmVtJpzR8){&?8{)s3aVBE_;j1|Q2qP(-tsa_YE4n$zqw-U+1>O{WMGlm<g;#;E2bjq+YhNk6v`G>Q}h=XWHUWE+t(pbZu$VwzF#>YK&isTK0W1%0|5SOXN4_t86yUu(0hqZ8<$HncO2pmbRnRt+hN&Jsa57RUpr-Uy(Wh|!@$%<Y{MR(`9mw$J_l!<JNcy_@(BQDqp^5`CNlOGVb4qBm#1nLiL>GVE{Kkz?Oa4r8PNqDyU$&C8>drHF%BJLgb_MHg+Yj5fsAj}Ub*3X|N(E~BG48+rYY2PG?Pa{HLsp%*^wa2WGEY@bp%M^DJM=DE;til*Lg1KVMm}0?&#e4Y&wG$19Oe1WK7@O~*fm3oiz{i1REHnT+bq(w#5j9Tu9Yqb~@RC1S+V4rS5zDbWO!df1I@(-3bu$>?jWrE};M)dO*XX<}>l_o@<1Pjfzn3rX$r3|OGm(q2VV4<42o?^bN;+kr+->;I{qDe+Z}Y83W6H3ZeyV*h|Dbj;0r~h<i;jfgS|s%dqm)V5+WXa?DLKPRNOnxTjl*ING!4ISFJIrer-6uHUMP_4nFL;*h!f+9M|WKYti54M4LOZK&Uc**l5l<hWOCZOA#RZeg!NN>w=^;aF%;j5;n~r0{KRo*P(ykS{mLjzOi!?}egdmENkX)q4jBVp7kZgyeo(T*B3WVu#5X-LG0mU2`W8H4zGs<!!-Y6`FaMwxzHt(adc+M|Lf-)G#7Z!(+J$i)Dt=r*daB5B(U=L`ehdX2-R0y>k_FXcm2h@z!Net_aEE0?7#W&8z%UHKSfK^NU;3eIIyvwgcNXt+|0YSRe$=7ZmmFc-ia9-#cF_f<abZPp49!pHhxz`&<9^1T{W){h`{cYyQVCbBBpKuQE(c!JZxCSFL%=<{VV!Wo_LDXv858h*i*@0knx#I|z5JUC;M~@0kB%=ZK&vJLtz;*F@c1=8>=;3L=hpaxZ|UHC7-*c}zL$SdiwiNgp)D6a3F*MVWEi&(C&+-2)(YEE(fsNUlmNKt3Tom9hxhXGE^Wq%6~DdW2Z+dbmt;)$gbm;q!_2SaUdneVm?H9K$7(BN?1!))%l{!s(h%Pt@CVsFf_zGE_<#6G$|jp`GG_EEUlo&*plYZXj0h5n`I)!kZ<3_vL;x5wLARb&Nm-UN{B!)$)SBkU=)RbcJD!0gva@33AkEy%t@txDBxlAW;mPT95W*_xl@<ffNbto!upWNX$P+#Zr<(Z2`G^&XJ$i1%^V{=79>NcyPuV?~$<|o#A3OTt%^*?vh97~(w?7+BC;A;DSS|aG*RA-wU+7sP;@K;}#*{|KB?f#H1h+Kb)NI&-C=ZQhh6(HL&KAm#{n)MeGcC+kblFU#=836;blUDXHLTf0waz)N*ERT2BKBuM`h2qzg`=Oo-HJa9pH8vf@Rf1-wT~pv8PQk+Ph9Zh1>YcN;R$DP8s=-i8P`Yxc%ij##oK#ygG=|rCEw)Yiwz1N>o*4}(zT1A69j%}9>Y5PxpByea9^OMdMp0kI2qA59T|a!<OGlngjg@UJCZRGXbG6dgX*EXzM(`$3w{=siO!O5#UE}bMNgvhY4`wZ1SaH6ecK5CI0+|%MazxZ_Zxf!796Hxb;#u&m&->GeN2){sDSPY2Fpq!(J?l(Ci$9ehJz?(iW_WJp^=Rd<o(Dc#l1q6kKO)5l3aAvjOCw`pyP)rn2L5bNOvMf75p${`6U4snDLnkwE?--HKy|znLjm2{_V3QnGr8s(7>sqjeg(Ah>S>{@3BG|^K?H`ijbe*jpMg^@Fp0N*t6f0yOn=<YI+6$03QE+V>>t0fIwH1AHyPM<O9MNX+)Ly?PnBz3hyWVinXry^6%}5+58Gku3f`abU!j<Au#F@p2_fZ99C-}!N_qE1N3P#qQiWT{gdXr_vDZtSUUS9(@iQM1Tbia#sE9)3t|xB1pSKyBf=PNvWy>jp!<CCz5IJm&I-lcs<Ea6>qeKOOA)KrfH_9}5Io@Zv{nfR_e9>A)0An%DBR0`S2z3zwi*XVswjA!<VP=XH++sHl}<EGNS^PZ(f!q`neR?|n(FzJr+*m>@x-ga9KIoS6o=uCWu#Wj)y!`pbY~Q+<f$Q~O>i{OaN;Ekb1&cC^I*T{;1Mp_Ay0PX#g$>@*;5zLQnr|PJ9VbZPZACTfITbEuFt*v!&>+&Cex;j47kW;<n5dn7AE->rCjMkfUXd`6Y!LL17akN*f-E!y_bJ@D<ZgU`^_{+6#|sJaE5?@C->9xj=<rdd9f%+2t}W(H_6jT?tXFkQvOYn1sbGc@|z_C^@;;3!Q>`uWB5LyuHrWAph|@Jt!jDlcOpzv+4Elhy^~%sv;%D?%VQ9cmtjnH@k<FrFjPb)Af?SAlN=dHV!*cI2U7gH!|F|v$QT`mx+&DiRy4TG0QuoVp`YF&AB)c-evm7xTj{kIcvFHC&v7sRpmuqWxA+|uvPoe^_4v)K%v9niDo;Dz8<78S27E3Lo)AH#im)no&WZnfC*89T_5v!wA5MuFc<1Cuaz%Rv2+Nfv_alD4`IJ3+=3sqC!sNaDgW6uj{5S#mKcE*Eo}+6IFJ;TUDbehd-N7DGh{9G3I6AQW-m&=B-mUzD+KTLEza$6rp_FY{X3z1kziL@$djA*pec)m$<jH%OD|qsK)_v#u?@6)<)Lu)8Ay4<raAAY_x8H;(8AE5sZyOMgr@=g!C-o%<!n~`W6Wq$bUV}#<(t(u%B{cDy`XPG+4lFN-h19{@A_3bWE1zgE>V(rqa-qDJfA@cIp(w01$>;bMQV)nf!h&}{2$&yFV#fkW1XsYPm5wCKj%XOma_{|{Br)XEYr2eA4CHg4DoiH&Fhsz}Z_xuYaEGO0ft`^x`^^#Y4Z(Z)_{MC3E%`@6L;NBkc@kpgrJA2L2j7jfC@ty1lLn}>D4$7cg10f1_ww)URy?%AD=;peA;XWNM6Cd|#BJDC|1pO{A#9Ifp^CV=`|4Jj8t=XTo+Quebu2#+Oo<sx<>2tPZw2^<1Ae9~IN_Ps5mxa1rp>oaoEm}tJ~?la1U8I^;%yNP8T^A9)k3z2-y@8W@?eA(lsR4M2XW6uf`i6jaMPXdzu#^h0CCn`M&*rX^`}$hqx)TovLPSLtKYd2r)w;U;{xAcB#Ho3K9>Jul0;t(RYPFSFZ?YO8^k9_6AL?h1D$o>%l)t+Eq)oE={#;iuvY0-{B>x0h!99cJUnrp@N$f^pj8deAxHs1z~jKe-{Xl1m2O7Vc=%QC#os$)xJ1A9f#l6ePARhdI6lx93h(7V(u96jn2uLuiTwf(1cHanc8c7JKNEv^F2Oae;6DrGm#JA^Ce%;(eYXnoZMo-BO5v3X>5KrmEFXAtx3O=<-`lPD{H4A;N<x|m*U=@B$0Ya{BlyqS9N&33esh_DVXS5f*auBAu223<3$97=^?@OSsz$`%5w`Yd$1ix=TK*d})Csrv#p1+pb=Rm$vN`2e{Jq^uJN)OA6t#p~p8KU}!%7Qff(OLn@x3-}HAZ|ZwqfAX@IL;t0xRB%Km1OHPRTD&Cc*)qBL3qKcos~Wbf~ayzvwHEGa*my#%%3%;>dnoJlL)H>&e{mc0b<7oUuR&$|ZG}?>a_UIUWCLsl5FR)ejViV`sIg<c9+Su5EdD3H>k$dy5Mt^_t(mvDJ?6$Ec^?s-`3PPeTNY-G9n}S!GFZZsI?P^r862B#H3&Ha)?nYLyh2Lwr)k<SuMMXpCYNJO(6#fJ2?Ed9x=^F1-FnFY|{a+2d;4&m@?Gm6{Jt&wc|VZFI!%J&}LONq*br*@WzQ`H$8?nZD8U%;2AzB>x3aO9KQH000080Qk6DU40oZNwy0B03R6u02KfL08>>^P)tchL|;QzUs6RuL`_95ZDfU5+in|25`CYqC`b<`1#-lTB+E4Ppo^UZNCL;Q6D$OS)0}CNy&|V4(><ga_+g*+12+G#@B0h;DfyB;RXxLtBw!85oLg5_SDiXl)yH&wb35n_UX!aPf7hw+&h;Ye+RD@IuOm|Qdu7r`npGy#QFKjdX`Sn?)WT%S8=I5UzA9*HOY(EQ754eJ8=9)cqMK<Cq5NSw$Gy&HI;I~y8Ap@KEIc_Y3tehU9Tla{OzNFHIyv#$`ID3IZbVw8b0}J5G{=9A*v3NJVxi^prOot$=E_m7SGt5@I`5Eyg}&5k>1e621^rc}ezB1iYZZQNIa9g$Lzm$j!tO}r8L|49KH&?|iVZ8aTP+|j+tl`BXJWrDp)`*EW-B~`f?Vs2l5{oZwJt6;3AOTSS!_h~{Jhw-gOWK!PR&BcQFMGP+E`H(B}o#U&h1j4s!7UDJ>}<ccXL0k){}9g;4~uqb9%Paxp(pWoc8{?xq@S90^N&+Y{tE%%u;SW+_qv55S)ilk?|tv`jqGff4BI%JKYMhdGH0?Ti}l`JG4+HlVB{&V+e(vPEBgy#X_ZzYNll`(KmX}Zu~~yHwa^dG`Ms7FRt%?eDOIiaMTh&WNrEued~6+2Y=9VjTM6q4e@i<p*Q&H4|>$Pcq0qwh97k3HGa-Jc-6c=zYOnh!;1_2T;P3=-}m3(eZBsUT35Rrx?*8GUeIMw)x(m0v*hA6FNsc0ew!<gToM==7X#DR*!fn7FBNdgXkh523sV4PrKZB#h0aEFdFZJQ9_)Ll6P>*dpSYOby^tjXsFL*Q>|(S7H(%9xvu+X!=1i=g)2y^r0R?QE*igUcdp4oDg|V5LPIU?NF*Kab5vPyuuJ3+&N2`9^Z^!J1`#Ke2PVmO)<Rpr^NcB(b0+Y`=l!!ORz20B{_;<J*oG18>1{X~#cSucmU;RNmq`B<E1tWZ&tgl>fDkK8W!K2O_pU#m7er;0C*>ITruV|)oT`F&p$81vrJJ^J{1dHV2jo1TdJhA629y~W^%cqA{z^q`$+Q82TSpyxX5)>r6#|}8EAeYTfl!QVRg;wA!n?nkxV-PJ)bQbc3Fu^V%T{xx#N^S1-S1<&91c_0?&Witza3#Xw>$$<2hB5JE8Ki7QOBf}O@8AOC4lS2#V19S|Vf?c{NCYukH~!f>OSTT$tqy7<JX|`xSZT)_31GhK>}=p9gXVoI-}fC%WLPtrn~kgiOcWrUKEe-b2E*Xp5PdT<xmvhUfZ@GbYASVV3e6!dHIsbCI{EAV<uhXge{&_785n$M4AVO5oblpY{03aTW5Yq46Aiu4QsqpjT0$(5l~<imSIi9NN!iK6juAo>5oC~OQsM6bc9?9-OyaZM21k!+=V^?B;jB1GO_{Ll>B1^M98mHwq*7%-Rfj%-LN6}AAWojnd-VMuZ$%EC0a|c#r2|RzFVA4o5;X7?Yf;~^(H!f4;LmgVr4C0>Tf2nUeSi1<oK~jv6^yD&n+u!b`pT;fN=%8Zz`)WjDcLJ;Nh~!~J`IQo)Fxg~>_}E5_;Cwt91-N*K8%p7qX~Akw&kN^=Txo|%%J%GBE63Ds`OHSi;bV#JTl9|mdM5pswV!63+3lA_NYtW0u+x<`N65Trzpmc#viqt6-{xDBEj(0_Cd@U)H=F8YSWW{#{^L5^;~nu^X&5umDc*Jokn>Lg+*y{-{QMx5}zq(kSl#i!zfKHVN<~*M(9GwA>^^i+~~QeS51<?qBxGFbNGsF$~ulr3xZxW=y(gchv61#Zx0tDQXQJKLvLw-0<XcJ5Ask{Z924Ww1sSXN#8MT=;70s5%sYpyw_ZMkTq|ySf2N|hnp_cY^Fnz18n-cFo-H7Cvev&3wO_z9f##3@?!xJQh2Q4<(6xA<ki@33f*axF*>U<)nkrk6iFcv`L@t`>!|T9KmwS~M+bV~lcmZxs2+J{*C7o$^r&?aY9P>0EJm!aKf0p%!x56;=nJPozt`)H`j@@l7hs>%)PsYXnkgP($X+*vbW0ml_vkfW?Eob2qtdR~O%JW3!1YHRsnXmQoyC3p^x`)|cmr_xvcI8LSU={l*WM%7_>c3LYgjwlX2iNQ9*zsPV7zdxBfhzgQ@4^dS-W;<JMXcQw^X@?p7$!gW+LN?-vJ9_Ldhj8fsmUniOLGjY#ktkK+DpDj)W*C%FUy|7PQy$18`IOfAaCgf5&wrQ-vPeMF+I-f}(DhiGdDzfuEwQ2eIy6?9UDXV5@#`u4M2FqQlGUtF3$KMy_w<`c|%Q<@!#p@8r50TAr2k79kLr`h=#M=`M7)&3s3#?QHVuDXN60dG<hw0Cu*6V13zq9qlzhVYJeXz@=6g2Ji3WIjG<HkZIBY2mMcb8Pu{z#!WqavV=$ypWgSmCi)-jXP~*ykRFQMUVJX?(T~DK(&dMGE)_PQ$4oI|Nw=N+j=E#~2HCQ(>x5^pgoj2qF7?XjH7eZu1IVFhM!5z_0Gp*=t1`nYpb)(uS_A+(YkFESxg2rJQ03?Wr0d<~inN?a>Hq#`K+=n|YhhznTC|k#D(1o{wG}$Wj6E-W$sO%NmCKm!E%!RySxebnYcrdBuqY&M`J=^VWO}M99+9V%)co8|+7t|Ohxrvp$&SLe3^@+Mui)8s@KEz&sY*0%V|RttGUTna4xx^Jgg`LWhVy3|T<%d+sU}Q}JfI^I_<&&m_p>p}DC)<+<$F_;o1fQI_`+1NMJCJQP=Dga#pLOt%Idx4+H;d3&;V77G^S(D<k08H<nKL?csVy~9K@2hIn~NnB{xmV8=yJ`11@YDPF%P?r<hbYn}FmF%Pg$(-J-Oqc8;y%oh~uYb0g>HrFJq5#`LosjX>$vq9^;47X2jIO*Y{*>v%fVci0lPM-UnKA{wv<e@4><;t61@<v&en_St7A#n^mhAXm!?ZytkcVbi(WWia>WtV*#F0T5k<Z75C+IrS05%EuGNlL%nFjZRD?A=tjkS!0H7ITKI6;JZ%v%E-~q@EIfRm7EaSqF^vVeK^QELvI)j*$W@)Ldgv9jL>+Wq2Xs5+TzFYRJl7^;l#YX|8@H$j?P%}y<l4Uh2$AkCoJuwmIjm!3N%~iLE}^i>`H-`$B5fG%l%m8%xqjT_|U*H8?m(SK78H*-?;o0x9IR#R;7mXsR+kyS)kprf@@aXgY$tC560}}Ufdh-;1~CXAkdgT1hbk`12~`Dl1~V^@PfF)L@l55w2)ilKoS>h=5L%;wrNS~5owo+ca{zG(2tOW^G7xpDaCY#(5;wt3!_=5k^{{qEN|J;CN=lx#O>jwkd+HG^PROB>v+uVdT)G0tIS+@7~XL5A@71MfT~|>9r`Rst-aGtq1^5KoU~<%dDI&|$xuR_lOR~!9ZU~%U)Z|zMlhj9q<k4p?IfBXxG;;H&K~UqG{fgb<;8wsPrJ?$B21TPkv2icue-C~6qzF)=X_#pya;u6<?eWdkzWtBJDi7;kY|HDHM)d2YlmX2uNgw<GY1bLZBvFwAT)4Yg&j|HO+tt|Y;A_``{r1YQHB0Ymt|e@WP;E4%62^yHJyR0VB6O)d9!S=McpQ&*22Ez6Ub-$p+6obo#tkMn{f0I?uT+;>iUcf{GtNDhFlhzii}W8OxOAT86Rxm10#I61u(bxAne+zR&rwzr}24xnDf^^|3&BR(BJHxZ)XA!E-PA2F5cD+i5!ruOAGp|^#oY1#g(W)><-OU1q;_GCVF?|;DTd6=yGX-PvBD<um;}c(L}HjhLWi=$W`Gk#a=jA0|If6^pl=vh7*ei7+DXSRLcA;gO4B+_uQc%6W>=b4n-HB3XJSrwJgMqBT6!xy3=~NKs&n3lBOah;b}KrB{V^|mkKsTxtypJHje%WP)h>@6aWAK2mtuFTV3SC(O)zG002M$000*N002`}P*6-sMMPgqNmO4`NI^0+HZF8{bOpf)0l+W_13*iG@&q|-6#o)>*K>y#)@()2rrANsw#6rf+kSTO)2Z+bqbQ`pI*AF<2T)4^0u%rg000080Qk6DT~VIrYcM<j042!)02crN00000000000Du7i0001CZewU~a$j^~ZeKGnUuAG_V`y_OaCuNm0Rj{N6aWAK2mtuFTV34c^KB_0006aM0015U0000000000004jiWjp`?bYpH`W^!+OWo}<^Z*z2Ta&Kd0b8}y1c42HTaCuNm0Rj{N6aWAK2mtuFTU}3?`bMJ<001~D000XB0000000000004ji%3J^dV|HI=Z){|9E^v8JO928D02BZK00;p1xLaK-W(QQG82|t}IRF3@00000000000001h0hVk40CRM5Wnpb!V`yb#YjbsNb1q|Zc2G+J0u%rg000080Qk6DU40oZNwy0B03R6u02KfL00000000000Du8&h5!IlRZvh&Nkv3oLsnl>ML|SOMJ{b*P)h*<6aW+e000O8__$kL<ipWlGynhqKmY&$7XSbN0000000000fB{sJ002`}P*6-sMMPgqNmO4`NI^0+HZF8{bWlqL1^@s600saC0F(g$0KJj`0000'
FOLDS = ('A', 'B', 'C', 'D')
TRAIN_IDS = tuple(f'train_{i:02d}' for i in range(26))
VAL_IDS = tuple(f'val_{i:02d}' for i in range(10))
SUBJECTS = {'A': ([2, 3], [4, 9]), 'B': ([1, 7], [5, 12]),
            'C': ([5, 6], [6, 10]), 'D': ([8, 9], [0, 11])}
FINGERPRINTS = {
    'A': ('5760f8093c6433c0', '27ae219e907ab9ea', 'd8deead0cfe9437a', '39b17a2c3e522d7b'),
    'B': ('9f52151329e5f589', 'cf5d4c3382ab73ec', 'dfb0ff6bb8c0c582', '10ba71a9ed553778'),
    'C': ('3531fd8cad97663e', '645d06a6f38a98df', '267822d1b7a72e49', 'e34d880f0840e366'),
    'D': ('80e902563ff366bc', '15824eccf42780b4', 'e45ab3c4a7f6eb94', 'd595eaf5e54fe1fa')}
CORE = POST = CV = np = h5py = torch = None


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    def convert(x):
        if np is not None and isinstance(x, np.generic):
            return x.item()
        if np is not None and isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def module_from_file(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def initialize(support):
    global CORE, POST, CV, np, h5py, torch
    raw = base64.b85decode(PAYLOAD)
    if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA:
        raise RuntimeError('Bundled code integrity check failed')
    support.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = (support / info.filename).resolve()
            if not path.is_relative_to(support.resolve()) or info.file_size > 2 * 1024**2:
                raise ValueError('Unsafe bundled path or size')
            value = archive.read(info)
            if path.exists():
                if path.read_bytes() != value:
                    raise FileExistsError(f'Bundled file has changed: {path}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as handle:
                    handle.write(value)
    # No yaml import: folds already exist, so the builder is never invoked.
    import numpy as _np
    import h5py as _h5py
    import torch as _torch
    np, h5py, torch = _np, _h5py, _torch
    CORE = module_from_file('_shared_cv_original_tcn', support / 'anchor_tcn_30_epochs.py')
    POST = module_from_file('_shared_cv_original_post', support / 'tcn_frozen_postprocess_eval.py')
    CV = module_from_file('_shared_cv_supplied_scorer', support / 'cv_folds.py')


def stream_digest(path):
    """Identical content hashing to the supplied builder's stream_sha256."""
    value = hashlib.sha256()
    with h5py.File(path, 'r') as handle:
        cir = handle['cir']
        value.update(repr((cir.shape, str(cir.dtype))).encode())
        for start in range(0, cir.shape[1], 20000):
            value.update(np.ascontiguousarray(cir[:, start:start + 20000, :], dtype=np.float32).tobytes())
        value.update(np.ascontiguousarray(handle['t'][:], dtype=np.float64).tobytes())
        value.update(np.ascontiguousarray(handle['radar_serials'][:], dtype=np.int32).tobytes())
    return value.hexdigest()


def inspect_fold(root, fold, cfg):
    folder = root / fold
    meta = read_json(folder / 'fold.json')
    fall, adl = SUBJECTS[fold]
    if meta.get('fold') != fold or sorted(meta.get('fall', [])) != fall or sorted(meta.get('adl', [])) != adl:
        raise ValueError(f'{fold}: subject split differs from the supplied guide')
    if meta.get('seed') != 2027 or meta.get('train_stitchings') != 2:
        raise ValueError(f'{fold}: expected data seed 2027 and two training arrangements')
    selected_stitchings = []
    for index, (role, stitching) in enumerate((('train', 0), ('train', 1), ('val', 0), ('val', 1))):
        matches = [s for s in meta['stitchings'] if s['role'] == role and s['stitching'] == stitching]
        if len(matches) != 1:
            raise ValueError(f'{fold}: missing or duplicated {role} arrangement {stitching}')
        entry = matches[0]
        count = 13 if role == 'train' else 5
        expected = [f'{role}_{i:02d}' for i in range(stitching * count, (stitching + 1) * count)]
        if entry['streams'] != expected or entry['fingerprint'] != FINGERPRINTS[fold][index]:
            raise ValueError(f'{fold}: {role} arrangement {stitching} does not match the reference plan')
        selected_stitchings.append(entry)
    train = CORE.discover(folder, 'train')
    available_val = CORE.discover(folder, 'val')
    if set(train) != set(TRAIN_IDS) or not set(VAL_IDS).issubset(available_val):
        raise ValueError(f'{fold}: need exactly train_00..25 and at least val_00..09')
    val = {sid: available_val[sid] for sid in VAL_IDS}
    selected = {**train, **val}
    times, records = CORE.inspect_streams(selected, cfg)
    for sid, path in selected.items():
        with h5py.File(path, 'r') as handle:
            n = len(times[sid])
            for start in (0, max(0, n // 2 - 2), max(0, n - 4)):
                if not np.isfinite(handle['cir'][:, start:start + 4, :]).all():
                    raise ValueError(f'{fold}/{sid}: inaccessible or non-finite CIR (check segments.h5)')
    events = CORE.load_events(folder)
    # All CSVs are filtered to the explicit selection; val_10..14 never enter scoring.
    unknown_train = {e['stream_id'] for e in events if e['stream_id'].startswith('train_')} - set(train)
    if unknown_train:
        raise ValueError(f'{fold}: labels mention unexpected training streams')
    summary = dict(fold=fold, source=str(folder), train_ids=list(train), validation_ids=list(val),
                   excluded_validation_ids=sorted(set(available_val) - set(val)),
                   heldout_fall_subjects=fall, heldout_adl_subjects=adl,
                   stitchings=selected_stitchings, original_metadata=meta, streams=records)
    print(f'[fold {fold}] 26 training, 10 selected validation; excluded {summary["excluded_validation_ids"]}', flush=True)
    return dict(folder=folder, train=train, val=val, times=times, summary=summary)


def create_scoring_view(view, fold, data):
    """Tiny private external links let the unchanged shared scorer see only selected data."""
    folder = view / fold
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    for sid, source in {**data['train'], **data['val']}.items():
        role, number = sid.rsplit('_', 1)
        destination = folder / 'streams' / role / f'stream_{number}.h5'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source, 'r') as original, h5py.File(destination, 'x') as wrapper:
            for key in original.keys():
                wrapper[key] = h5py.ExternalLink(str(source.resolve()), '/' + key)
    selected = set(data['train']) | set(data['val'])
    for name in ('streams_events.csv', 'streams_composition.csv'):
        rows = read_csv(data['folder'] / name)
        save_csv(folder / name, [r for r in rows if r['stream_id'] in selected], list(rows[0]))
    shutil.copyfile(data['folder'] / 'groups.csv', folder / 'groups.csv')
    meta = dict(data['summary']['original_metadata'], heldout_stitchings=2,
                stitchings=data['summary']['stitchings'])
    save_json(folder / 'fold.json', meta)
    save_json(folder / 'VIEW_ONLY.json', dict(
        source=str(data['folder']), note='External links only. No streams rebuilt. Only val_00..09 selected.'))


def verify_reference(out, view, data, support):
    reference = {(r['fold'], r['file']): r['sha256'] for r in read_csv(support / 'stream_checksums.csv')}
    rows = []
    for fold in FOLDS:
        for sid, source in {**data[fold]['train'], **data[fold]['val']}.items():
            role, number = sid.rsplit('_', 1)
            name = f'streams/{role}/stream_{number}.h5'
            print(f'[verify content] {fold}/{sid}', flush=True)
            got = stream_digest(source)
            rows.append(dict(fold=fold, file=name, sha256=got, reference=reference.get((fold, name)),
                             identical=got == reference.get((fold, name))))
        for name in ('streams_events.csv', 'streams_composition.csv', 'groups.csv'):
            got = digest(view / fold / name)
            rows.append(dict(fold=fold, file=name, sha256=got, reference=reference.get((fold, name)),
                             identical=got == reference.get((fold, name))))
    save_csv(out / 'reference_verification.csv', rows)
    if len(rows) != 156 or not all(r['identical'] for r in rows):
        bad = [f'{r["fold"]}/{r["file"]}' for r in rows if not r['identical']]
        raise ValueError(f'Selected data differs from the supplied reference; training stopped: {bad}')
    save_json(out / 'reference_verification.json', dict(status='PASS', files=156, streams=144,
        metadata_files=12, all_identical=True, selected_validation_ids=list(VAL_IDS),
        note='Signal hashes and selected CSVs match the default reference; extra validation arrangements excluded.'))


def training_device(args):
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('This Python cannot use CUDA; use the existing server CUDA Python')
        index = device.index if device.index is not None else torch.cuda.current_device()
        device = torch.device('cuda', index)
        free, total = torch.cuda.mem_get_info(device)
        if free < args.min_free_gpu_gib * 1024**3:
            raise RuntimeError(f'Only {free/1024**3:.2f} GiB GPU memory free. Need '
                               f'{args.min_free_gpu_gib:.2f} GiB; rerun when the GPU is available.')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        print(f'[GPU] {torch.cuda.get_device_name(device)}; {free/1024**3:.2f} GiB free', flush=True)
    elif device.type != 'cpu':
        raise ValueError('Use cuda:N or cpu')
    return device


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_model(dataset, cfg, device, out, plan, policy, swa_start):
    CORE.seed_everything(cfg.seed)
    model = CORE.AnchorTCN(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    if params != 774339:
        raise ValueError(f'Original TCN parameter count changed: {params}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    averaged = torch.optim.swa_utils.AveragedModel(model, device='cpu') if policy == 'swa' else None
    rng = np.random.default_rng(cfg.seed)
    epochs, updates, skipped, global_batch = [], 0, 0, 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    began = time.perf_counter()
    fields = ['epoch', 'batch', 'global_batch', 'batch_samples', 'loss', 'classification', 'localization', 'optimizer_update_applied']
    with (out / 'training_log.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            start = time.perf_counter()
            seen = np.zeros(len(dataset), dtype=np.uint8)
            order_hash = hashlib.sha256()
            total_loss = cls_total = loc_total = 0.0
            count = positives = valid_count = epoch_updates = epoch_skips = 0
            for step, indices in enumerate(CORE.epoch_batches(len(dataset), cfg.batch_size, rng), 1):
                if seen[indices].any():
                    raise RuntimeError('A training window repeated within a full epoch')
                seen[indices] = 1
                order_hash.update(indices.astype('<i8').tobytes())
                x, y, valid, left, right = [v.to(device) for v in dataset.batch(indices)]
                positives += int((y * valid).sum().item())
                valid_count += int(valid.sum().item())
                optimizer.zero_grad(set_to_none=True)
                with CORE.amp_context(device):
                    logits, distances = model(x)
                    a, b = cfg.context_frames, cfg.context_frames + cfg.core_frames
                    loss, cls, loc = CORE.detection_loss(logits[:, a:b], distances[:, a:b], y, valid, left, right, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite training loss: epoch {epoch}, batch {step}')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if device.type == 'cpu' and not torch.isfinite(grad):
                    raise FloatingPointError('Non-finite CPU gradient')
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                applied = scaler.get_scale() >= old_scale
                epoch_updates += int(applied)
                epoch_skips += int(not applied)
                if applied and averaged is not None and epoch >= swa_start:
                    averaged.update_parameters(model)
                n = len(indices)
                lv, cv, rv = (float(v.detach().cpu()) for v in (loss, cls, loc))
                count += n
                total_loss += lv * n
                cls_total += cv * n
                loc_total += rv * n
                global_batch += 1
                writer.writerow(dict(epoch=epoch, batch=step, global_batch=global_batch,
                    batch_samples=n, loss=lv, classification=cv, localization=rv,
                    optimizer_update_applied=bool(applied)))
                if step % 250 == 0:
                    handle.flush()
                    print(f'  epoch {epoch}/{cfg.epochs}: batch {step}/{plan["steps_per_full_epoch"]}', flush=True)
            if count != len(dataset) or not np.all(seen == 1):
                raise RuntimeError('Full epoch coverage audit failed')
            if positives != plan['positive_reference_exposures_per_epoch'] or valid_count != plan['valid_reference_exposures_per_epoch']:
                raise RuntimeError('Extracted labels disagree with the training manifest')
            sync(device)
            updates += epoch_updates
            skipped += epoch_skips
            row = dict(epoch=epoch, samples_seen=count, expected_windows=len(dataset),
                every_window_seen_exactly_once=True, order_sha256=order_hash.hexdigest(),
                mean_loss_per_sample=total_loss / count, classification=cls_total / count,
                localization=loc_total / count, optimizer_updates=epoch_updates,
                amp_skipped_steps=epoch_skips, valid_frames=valid_count, positive_valid_frames=positives,
                seconds=time.perf_counter() - start)
            epochs.append(row)
            save_json(out / 'epoch_coverage.json', epochs)
            save_csv(out / 'epoch_metrics.csv', epochs)
            handle.flush()
            print(f'  epoch {epoch}/{cfg.epochs}: {count:,}/{len(dataset):,} windows; loss={total_loss/count:.5f}', flush=True)
    sync(device)
    elapsed = time.perf_counter() - began
    def checkpoint(chosen, path, chosen_policy):
        torch.save(dict(model={k: v.detach().cpu() for k, v in chosen.state_dict().items()},
            mean=dataset.mean, scale=dataset.scale, config=asdict(cfg), window_plan=plan,
            completed_epochs=cfg.epochs, checkpoint_policy=chosen_policy), path)
    checkpoint(model, out / 'last.pt', 'final')
    averaged_updates = int(averaged.n_averaged.item()) if averaged is not None else 0
    if averaged is not None:
        if not averaged_updates:
            raise RuntimeError('No successful optimizer updates entered SWA')
        model.load_state_dict(averaged.module.state_dict())
        checkpoint(model, out / 'swa.pt', 'swa')
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None
    summary = dict(parameter_count=params, full_epochs=cfg.epochs, seed=cfg.seed,
        training_windows=len(dataset), training_draws=len(dataset) * cfg.epochs,
        global_steps=global_batch, optimizer_updates=updates, amp_skipped_steps=skipped,
        checkpoint_policy=policy, checkpoint='swa.pt' if policy == 'swa' else 'last.pt',
        swa_start_epoch=swa_start if policy == 'swa' else None, swa_updates=averaged_updates,
        training_seconds=elapsed, peak_gpu_allocated_gib=peak,
        selection='Fixed final policy. No validation checkpoint selection.')
    save_json(out / 'training_summary.json', summary)
    model.eval()
    return model, summary


def completed_job(folder, protocol_sha):
    for attempt in sorted(folder.glob('attempt_*'), reverse=True):
        marker = attempt / 'COMPLETE.json'
        if marker.is_file():
            info = read_json(marker)
            if info['protocol_sha256'] != protocol_sha:
                raise ValueError(f'Completed job belongs to a different protocol: {attempt}')
            for relative, expected in info['artifact_sha256'].items():
                if not (attempt / relative).is_file() or digest(attempt / relative) != expected:
                    raise ValueError(f'Completed artifact changed or missing: {attempt / relative}')
            return attempt
    return None


def run_fold(out, fold, data, args, cfg, device, protocol_sha):
    fold_out = out / 'folds' / fold
    fold_out.mkdir(parents=True, exist_ok=True)
    chosen = {seed: completed_job(fold_out / f'seed_{seed}', protocol_sha) for seed in args.seeds}
    missing = [seed for seed, path in chosen.items() if path is None]
    if not missing:
        print(f'[reuse] Fold {fold}: all completed seed jobs verified', flush=True)
        return chosen
    train_times = {sid: data['times'][sid] for sid in data['train']}
    targets, eligible = CORE.build_targets(train_times, CORE.load_events(data['folder']), cfg)
    manifest, plan = CORE.build_window_manifest(train_times, targets, 80, cfg)
    save_csv(fold_out / 'window_manifest.csv', manifest)
    save_json(fold_out / 'window_plan.json', plan)
    save_json(fold_out / 'dataset_summary.json', dict(eligible_training_events=len(eligible),
        train_streams=26, selected_validation_streams=10, plan=plan))
    cache = fold_out / ('ema_cache_' + stamp())
    cache.mkdir(exist_ok=False)
    selected_paths = {**data['train'], **data['val']}
    cache_bytes = sum(3 * len(t) * 368 * 4 + 128 for t in data['times'].values())
    needed = cache_bytes + max(1024**3, int(cache_bytes * .1))
    if shutil.disk_usage(out).free < needed:
        raise OSError(f'Need {needed/1024**3:.2f} GiB free for one fold cache and outputs')
    save_json(cache / 'OWNED_BY_THIS_RUN.json', dict(run=str(out), fold=fold, protocol_sha256=protocol_sha))
    store = None
    success = False
    try:
        # Recheck availability before preprocessing, in case another user's job started.
        if device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(device)
            if free < args.min_free_gpu_gib * 1024**3:
                raise RuntimeError('GPU memory is currently occupied; retry this run with --resume later')
        print(f'[preprocess {fold}] continuous background removal for all 26 train + selected 10 val streams', flush=True)
        paths = CORE.prepare_ema(selected_paths, cache, cfg)
        store = CORE.Store(paths, 'ema')
        norm_out = cache / 'normalization'
        norm_out.mkdir()
        mean, scale = CORE.fit_normalization(store, train_times, cfg, norm_out)
        for seed in missing:
            seed_cfg = CORE.Config(**dict(asdict(cfg), seed=seed))
            job = fold_out / f'seed_{seed}' / ('attempt_' + stamp())
            job.mkdir(parents=True, exist_ok=False)
            for name in ('normalization.npz', 'normalization_summary.json'):
                shutil.copyfile(norm_out / name, job / name)
            save_json(job / 'config.json', asdict(seed_cfg))
            print(f'[train] Fold {fold}, seed {seed}: {cfg.epochs} full epochs, {len(manifest):,} windows/epoch', flush=True)
            dataset = CORE.FullWindowDataset(store, train_times, targets, mean, scale, manifest, seed_cfg)
            model, summary = train_model(dataset, seed_cfg, device, job, plan,
                                         args.checkpoint_policy, args.swa_start_epoch)
            inference = job / 'inference'
            inference.mkdir()
            sync(device)
            started = time.perf_counter()
            proposals = []
            for sid in VAL_IDS:
                print(f'[infer] {fold}, seed {seed}, {sid}', flush=True)
                values = CORE.infer_standard(model, store, sid, data['times'][sid], mean, scale, seed_cfg, device)
                t = data['times'][sid]
                np.savez_compressed(inference / f'{sid}_frame_predictions.npz', t=t,
                    confidence=values[:, 0], d_start=values[:, 1], d_end=values[:, 2])
                proposals.extend(CORE.decode(sid, t, values, seed_cfg))
            sync(device)
            total_frames = sum(len(data['times'][sid]) for sid in VAL_IDS)
            seconds = time.perf_counter() - started
            save_json(inference / 'proposals.json', proposals)
            save_json(inference / 'summary.json', dict(seconds=seconds, frames=total_frames,
                signal_seconds=total_frames / 50, real_time_factor=seconds / (total_frames / 50),
                validation_ids=list(VAL_IDS), input_frames=800, retained_core_frames=292,
                inference_hop_frames=292, proposals=len(proposals),
                note='Measured offline inference including NPZ saving. No measured alarm timestamps.'))
            del model, dataset
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            artifacts = [job / summary['checkpoint'], job / 'normalization.npz',
                         job / 'training_summary.json', *sorted(inference.glob('*'))]
            save_json(job / 'COMPLETE.json', dict(protocol_sha256=protocol_sha,
                artifact_sha256={str(p.relative_to(job)): digest(p) for p in artifacts}))
            chosen[seed] = job
        success = True
    finally:
        if store is not None:
            store.close()
        # Remove only this exact newly-created, marker-owned cache on full success.
        if success and not args.keep_cache:
            resolved = cache.resolve()
            marker = read_json(resolved / 'OWNED_BY_THIS_RUN.json')
            if resolved.parent != fold_out.resolve() or marker['run'] != str(out) or marker['protocol_sha256'] != protocol_sha:
                raise RuntimeError('Cache ownership check failed; leaving cache untouched')
            shutil.rmtree(resolved)
    return chosen


def load_seed_data(jobs, seed, folds_data, evaluator):
    frames, paths, ground, durations, proposals = {}, {}, [], {}, []
    for fold in FOLDS:
        for sid in VAL_IDS:
            key = f'{fold}/{sid}'
            with np.load(jobs[fold][seed] / 'inference' / f'{sid}_frame_predictions.npz', allow_pickle=False) as z:
                frames[key] = {k: z[k].copy() for k in ('t', 'confidence', 'd_start', 'd_end')}
            paths[key] = folds_data[fold]['val'][sid]
            durations[key] = len(frames[key]['t']) / 50
        local = evaluator.load_ground_truth(folds_data[fold]['folder'] / 'streams_events.csv', ignore_flagged=True)
        ground += [dict(e, stream_id=f'{fold}/{e["stream_id"]}') for e in local if e['stream_id'] in VAL_IDS]
        proposals += [dict(p, stream_id=f'{fold}/{p["stream_id"]}')
                      for p in read_json(jobs[fold][seed] / 'inference' / 'proposals.json')]
    if set(frames) != {f'{f}/{sid}' for f in FOLDS for sid in VAL_IDS}:
        raise RuntimeError('Inference coverage is incomplete')
    return frames, paths, ground, durations, proposals


def subset(items, folds):
    return [p for p in items if p['stream_id'].split('/')[0] in folds]


def select_decoder(evaluator, ground, durations, frames, proposals, refiner, tuning_folds, folder):
    """Original staged search, using only the specified OTHER folds' labels."""
    folder.mkdir(parents=True, exist_ok=False)
    tune_ground = subset(ground, tuning_folds)
    tune_d = {sid: value for sid, value in durations.items() if sid.split('/')[0] in tuning_folds}
    tune_props = subset(proposals, tuning_folds)
    trace, best = [], None
    for nms in POST.NMSES:
        kept = POST.suppress(tune_props, nms)
        for threshold in POST.SCORES:
            pred = [p for p in kept if p['score'] >= threshold]
            report = evaluator.evaluate(tune_ground, pred, tune_d)
            trace.append(dict(threshold=threshold, nms=nms, mean_f1=report['ranking_score']['f1_mean']))
            if best is None or POST.ranking(report) > POST.ranking(best[0]):
                best = (report, threshold, nms)
    save_csv(folder / 'control_sweep.csv', trace)
    fixed = POST.config()
    tuned = POST.config(threshold=best[1], anchor_nms=best[2])
    configs = {
        'tcn_fixed': fixed,
        'tcn_tuned': tuned,
        'tcn_fixed_refined': dict(fixed, refine=True),
        'tcn_fixed_refined_gate': dict(fixed, refine=True, gate=True),
        'tcn_tuned_refined': dict(tuned, refine=True),
        'tcn_tuned_refined_gate': dict(tuned, refine=True, gate=True),
        'tcn_tuned_refined_gate_nms': dict(tuned, refine=True, gate=True, final_iou=.1)}
    tune_frames = {sid: value for sid, value in frames.items() if sid in tune_d}
    run_trace, run_best = [], None
    for threshold in (.1, .3, .5, .7, .9):
        cfg = POST.config(candidates='runs', threshold=threshold, refine=True, gate=True, merge=True)
        candidates = POST.execute_config(cfg, tune_frames, tune_props, refiner)
        for nms in (.1, .3, .5):
            pred = POST.suppress(candidates, nms)
            report = evaluator.evaluate(tune_ground, pred, tune_d)
            run_trace.append(dict(threshold=threshold, nms=nms, mean_f1=report['ranking_score']['f1_mean']))
            if run_best is None or POST.ranking(report) > POST.ranking(run_best[0]):
                run_best = (report, dict(cfg, final_iou=nms))
    configs['signal_runs_tuned'] = run_best[1]
    save_csv(folder / 'signal_candidate_sweep.csv', run_trace)
    reports, rows = {}, []
    for name, cfg in configs.items():
        pred = POST.execute_config(cfg, tune_frames, tune_props, refiner)
        report = evaluator.evaluate(tune_ground, pred, tune_d)
        reports[name] = report
        rows.append(POST.summary_row(name, report))
    selected = max(configs, key=lambda name: POST.ranking(reports[name]))
    save_csv(folder / 'tuning_method_comparison.csv', rows)
    save_json(folder / 'tuning_official_reports.json', reports)
    save_json(folder / 'selection.json', dict(selected_method=selected, config=configs[selected],
        all_method_configs=configs, tuning_folds=list(tuning_folds),
        tuning_score=reports[selected]['ranking_score'],
        note='Stagewise historical search, with labels restricted to the listed tuning folds. '
             'Fixed signal constants; no 5.9-second center suppression.'))
    return selected, configs


def metric_rows(report, **identity):
    rows = []
    for theta, metrics in report['by_theta'].items():
        row = dict(identity, tiou=float(theta), mean_f1=report['ranking_score']['f1_mean'])
        row.update({k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool)) or v is None})
        rows.append(row)
    return rows


def score_prediction(folder, name, pred, ground, durations, evaluator):
    path = folder / f'{name}_submission.csv'
    save_csv(path, pred, ['stream_id', 't_start', 't_end', 'score'])
    checked = evaluator.validate_predictions(path, durations)
    report = evaluator.evaluate(ground, checked, durations)
    expected = evaluator.evaluate(ground, pred, durations)
    if report != expected:
        raise RuntimeError('Official CSV round-trip changed the score')
    save_json(folder / f'{name}_official_report.json', report)
    return checked, report


def evaluate_seed(out, seed, jobs, data, evaluator, annotate, view, release):
    seed_out = out / f'seed_{seed}'
    seed_out.mkdir()
    frames, paths, ground, durations, proposals = load_seed_data(jobs, seed, data, evaluator)
    refiner = POST.SignalRefiner(paths, annotate)
    by_method, flat_rows, selections = {}, [], {}
    all_reports = {}
    began = time.perf_counter()
    for target in FOLDS:
        tuning = tuple(f for f in FOLDS if f != target)
        print(f'[nested selection seed {seed}] tune {",".join(tuning)}; apply to {target}', flush=True)
        selected, configs = select_decoder(evaluator, ground, durations, frames, proposals, refiner,
                                           tuning, seed_out / f'tune_for_{target}')
        selections[target] = dict(method=selected, tuning_folds=list(tuning), config=configs[selected])
        target_out = seed_out / target
        target_out.mkdir()
        target_d = {sid: d for sid, d in durations.items() if sid.startswith(target + '/')}
        target_g = subset(ground, (target,))
        target_props = subset(proposals, (target,))
        target_frames = {sid: v for sid, v in frames.items() if sid in target_d}
        local_reports = {}
        for name, cfg in configs.items():
            pred = POST.execute_config(cfg, target_frames, target_props, refiner)
            checked, report = score_prediction(target_out, name, pred, target_g, target_d, evaluator)
            by_method.setdefault(name, []).extend(checked)
            local_reports[name] = report
            flat_rows += metric_rows(report, seed=seed, scope='fold', fold=target, method=name)
            for stitching in range(2):
                sub_d = {f'{target}/val_{i:02d}': durations[f'{target}/val_{i:02d}']
                         for i in range(stitching * 5, (stitching + 1) * 5)}
                sub_report = evaluator.evaluate(target_g, checked, sub_d)
                save_json(target_out / f'{name}_stitching_{stitching}.json', sub_report)
                flat_rows += metric_rows(sub_report, seed=seed, scope='stitching', fold=target,
                                         stitching=stitching, method=name)
            for sid, duration in target_d.items():
                sub_report = evaluator.evaluate(target_g, checked, {sid: duration})
                save_json(target_out / f'{name}_{sid.split("/")[1]}_report.json', sub_report)
                flat_rows += metric_rows(sub_report, seed=seed, scope='stream', fold=target,
                                         stream_id=sid, method=name)
        # Pick the target predictions with the OTHER-fold choice, never target F1.
        selected_pred = subset(by_method[selected], (target,))
        by_method.setdefault('selected', []).extend(selected_pred)
        selected_report = local_reports[selected]
        local_reports['selected'] = selected_report
        flat_rows += metric_rows(selected_report, seed=seed, scope='fold', fold=target, method='selected')
        save_json(target_out / 'selected_official_report.json', selected_report)
        for stitching in range(2):
            sub_d = {f'{target}/val_{i:02d}': durations[f'{target}/val_{i:02d}']
                     for i in range(stitching * 5, (stitching + 1) * 5)}
            report = evaluator.evaluate(target_g, selected_pred, sub_d)
            save_json(target_out / f'selected_stitching_{stitching}.json', report)
            flat_rows += metric_rows(report, seed=seed, scope='stitching', fold=target,
                                     stitching=stitching, method='selected')
        for sid, duration in target_d.items():
            report = evaluator.evaluate(target_g, selected_pred, {sid: duration})
            save_json(target_out / f'selected_{sid.split("/")[1]}_report.json', report)
            flat_rows += metric_rows(report, seed=seed, scope='stream', fold=target,
                                     stream_id=sid, method='selected')
        # These plain ids are consumed by the supplied scorer using the private first-10 view.
        local_pred = [dict(p, stream_id=p['stream_id'].split('/')[1]) for p in selected_pred]
        save_csv(target_out / 'selected_submission.csv', local_pred, ['stream_id', 't_start', 't_end', 'score'])
        all_reports[target] = local_reports
        print(f'[held-out {target}, seed {seed}] selected={selected}; mean F1={selected_report["ranking_score"]["f1_mean"]:.4f}', flush=True)
    pooled = {}
    method_summary = {}
    for name, pred in by_method.items():
        report = evaluator.evaluate(ground, pred, durations)
        pooled[name] = report
        save_json(seed_out / f'{name}_pooled_official_report.json', report)
        save_csv(seed_out / f'{name}_pooled_submission.csv', pred, ['stream_id', 't_start', 't_end', 'score'])
        flat_rows += metric_rows(report, seed=seed, scope='pooled', fold='ALL', method=name)
        fold_scores = {f: all_reports[f][name]['ranking_score']['f1_mean'] for f in FOLDS}
        method_summary[name] = dict(pooled_mean_f1=report['ranking_score']['f1_mean'],
            fold_scores=fold_scores, fold_mean=statistics.mean(fold_scores.values()),
            fold_sd=statistics.stdev(fold_scores.values()), worst_fold=min(fold_scores, key=fold_scores.get),
            worst_fold_f1=min(fold_scores.values()))
    crosscheck = CV.score({fold: seed_out / fold / 'selected_submission.csv' for fold in FOLDS},
                         root=view, release=release, verbose=True)
    if abs(crosscheck['pooled']['score'] - pooled['selected']['ranking_score']['f1_mean']) > 1e-12:
        raise RuntimeError('Supplied CV scorer disagrees with pooled selected result')
    for fold in FOLDS:
        if abs(crosscheck['per_fold'][fold]['score'] - all_reports[fold]['selected']['ranking_score']['f1_mean']) > 1e-12:
            raise RuntimeError(f'Supplied CV scorer disagrees on fold {fold}')
    save_json(seed_out / 'supplied_cv_scorer_report.json', crosscheck)
    save_json(seed_out / 'method_summary.json', method_summary)
    save_json(seed_out / 'nested_selections.json', selections)
    # Separate settings for a future all-data model. Their tuning score is NOT the CV estimate.
    print(f'[deployment settings seed {seed}] tune on A,B,C,D; saved separately from held-out scores', flush=True)
    select_decoder(evaluator, ground, durations, frames, proposals, refiner, FOLDS,
                   seed_out / 'all_folds_deployment_tuning')
    save_json(seed_out / 'postprocess_audit.json', dict(seconds=time.perf_counter() - began,
        component_seconds=refiner.seconds, joins_per_stream={k: len(v) for k, v in refiner.joins.items()},
        distinct_signal_spans=len(refiner.cache), profile_gate=POST.PROFILE_GATE,
        note='Offline processing. Fixed signal constants from the earlier approach. No measured alarm latency.'))
    save_csv(seed_out / 'metrics_all.csv', flat_rows)
    return dict(seed=seed, methods=method_summary, nested_selections=selections), flat_rows


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + f'_{os.getpid()}'


def package(out):
    destination = out / 'RESULTS_TO_SHARE.zip'
    temporary = out / 'RESULTS_TO_SHARE.partial.zip'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path == destination or path == temporary:
                continue
            if path.suffix.lower() not in {'.json', '.csv', '.md', '.txt', '.log', '.py'}:
                continue
            if any(part.startswith('ema_cache_') for part in path.relative_to(out).parts):
                continue
            archive.write(path, str(path.relative_to(out)))
    os.replace(temporary, destination)
    return destination


class Tee:
    def __init__(self, terminal, handle):
        self.terminal, self.handle = terminal, handle
    def write(self, value):
        self.terminal.write(value)
        self.handle.write(value)
    def flush(self):
        self.terminal.flush()
        self.handle.flush()


def summarize(out, seeds, all_rows, jobs, args):
    save_csv(out / 'metrics_all.csv', all_rows)
    save_json(out / 'seed_results.json', seeds)
    summary_rows = []
    for method in seeds[0]['methods']:
        values = [s['methods'][method]['pooled_mean_f1'] for s in seeds]
        row = dict(method=method, seeds=len(values), mean_pooled_f1=statistics.mean(values),
                   sd_pooled_f1=statistics.stdev(values) if len(values) > 1 else 0,
                   min_pooled_f1=min(values), max_pooled_f1=max(values))
        for fold in FOLDS:
            row[f'fold_{fold}_mean_f1'] = statistics.mean(s['methods'][method]['fold_scores'][fold] for s in seeds)
        row['worst_fold_mean_f1'] = min(row[f'fold_{f}_mean_f1'] for f in FOLDS)
        summary_rows.append(row)
    save_csv(out / 'seed_summary.csv', summary_rows)
    cost_rows = []
    for fold in FOLDS:
        for seed in args.seeds:
            path = jobs[fold][seed]
            training = read_json(path / 'training_summary.json')
            inference = read_json(path / 'inference' / 'summary.json')
            cost_rows.append(dict(training, fold=fold, seed=seed,
                                 inference_seconds=inference['seconds'], inference_frames=inference['frames'],
                                 inference_real_time_factor=inference['real_time_factor'], job=str(path)))
    save_csv(out / 'runtime_summary.csv', cost_rows)
    text = ['# TCN subject-held-out CV results', '',
        f'Four folds, seeds {args.seeds}, {args.epochs} full epochs each; checkpoint policy: {args.checkpoint_policy}.',
        'Training uses all 26 training streams. Evaluation uses val_00 through val_09 only.',
        'No checkpoint is selected on held-out scores. For each fold, decoder/method selection uses the other three folds.',
        'The selected CV row may use a different method for each target fold, chosen without that target score.', '',
        '| Method | Mean pooled F1 across seeds | SD | A mean | B mean | C mean | D mean |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary_rows:
        text.append(f'| {row["method"]} | {row["mean_pooled_f1"]:.6f} | {row["sd_pooled_f1"]:.6f} | ' +
                    ' | '.join(f'{row[f"fold_{f}_mean_f1"]:.6f}' for f in FOLDS) + ' |')
    text += ['', 'The ranking score averages event F1 at tIoU 0.1, 0.3 and 0.5. tIoU 0.7 is diagnostic.',
        'metrics_all.csv contains the available official precision, recall, F1, TP/FP/FN, boundary errors and false alarms/hour.',
        'Full official JSON reports preserve every evaluator field, including null values when a metric is unavailable.',
        'runtime_summary.csv contains measured training/inference times, update counts and peak GPU allocation.',
        'Epoch loss and exact window coverage are recorded for each run. Training loss is not an event F1 score.',
        'all_folds_deployment_tuning is for a future final model; its fitted score is not an independent CV estimate.',
        'This run evaluates CV only. It does not retrain a final all-data ensemble or produce a hidden-test score.',
        'Historical official-validation results helped choose this recipe, so fold D is not previously untouched.',
        'No alarm timestamp was measured; evaluator latency fields must not be described as measured inference latency.', '']
    (out / 'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folds-root', type=Path, default=Path('/home/abc2027/dataset/abc2027_uwb_cv_folds'))
    parser.add_argument('--release', type=Path, default=Path('/home/abc2027/dataset/abc2027_uwb_train_v1'))
    parser.add_argument('--output-parent', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2027, 2028])
    parser.add_argument('--checkpoint-policy', choices=['final', 'swa'], default='final')
    parser.add_argument('--swa-start-epoch', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.5)
    parser.add_argument('--min-free-gpu-gib', type=float, default=2.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--keep-cache', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error('Need positive epochs/threads and at least two distinct seeds')
    if args.checkpoint_policy == 'swa' and not 1 <= args.swa_start_epoch <= args.epochs:
        parser.error('SWA start epoch must be between 1 and the fixed epoch budget')
    if not .05 <= args.gpu_memory_fraction <= .5 or args.min_free_gpu_gib < 1:
        parser.error('GPU fraction must be 0.05..0.5 and free-memory requirement at least 1 GiB')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.folds_root = args.folds_root.expanduser().resolve()
    args.release = args.release.expanduser().resolve()
    parent = args.output_parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    out = args.resume.expanduser().resolve() if args.resume else parent / ('tcn_shared_cv_' + stamp())
    # Never place outputs in either shared data tree, including symlink aliases.
    for source in (args.folds_root, args.release):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('The output must be separate from both shared data trees')
    if args.resume:
        if not (out / 'protocol.json').is_file():
            raise ValueError('--resume must point to an experiment created by this file')
    else:
        out.mkdir(exist_ok=False)
    console = sys.stdout
    log = (out / ('console_' + stamp() + '.log')).open('x', encoding='utf-8', buffering=1)
    sys.stdout = Tee(console, log)
    try:
        print(f'[output] {out}\n[Python] {sys.executable}', flush=True)
        support = out / 'bundled_code'
        initialize(support)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        evaluator, evaluator_info = CORE.load_official_evaluator(args.release)
        annotate, annotate_hash = POST.official_module(args.release, 'annotate')
        required = ('_profiles', '_state', '_suppress_spikes', '_smooth', '_transition_bounds')
        if any(not callable(getattr(annotate, key, None)) for key in required):
            raise RuntimeError('The official annotation helper API differs; training has not started')
        cfg = CORE.Config(epochs=args.epochs)
        data = {fold: inspect_fold(args.folds_root, fold, cfg) for fold in FOLDS}
        protocol = dict(version=VERSION, payload_sha256=PAYLOAD_SHA,
            script_sha256=digest(Path(__file__)), release=str(args.release), folds_root=str(args.folds_root),
            config=asdict(cfg), seeds=args.seeds, checkpoint_policy=args.checkpoint_policy,
            swa_start_epoch=args.swa_start_epoch if args.checkpoint_policy == 'swa' else None,
            training_hop_frames=80, inference_hop_frames=292, selected_validation_ids=list(VAL_IDS),
            device=args.device, threads=args.threads,
            train_ids=list(TRAIN_IDS), evaluator=evaluator_info, annotate_sha256=annotate_hash,
            fold_metadata={f: digest(data[f]['folder'] / 'fold.json') for f in FOLDS},
            source_tables={f: {name: digest(data[f]['folder'] / name) for name in
                ('streams_events.csv', 'streams_composition.csv', 'groups.csv')} for f in FOLDS},
            normalization='Original 16 evenly spaced blocks of 64 training frames per stream; train only.',
            training_budget_reason='Five is a predeclared budget based on previous runs, not a CV-selected optimum.',
            selection='No held-out checkpoint selection. Nested historical decoder/postprocess search on other three folds.',
            python=sys.version, torch=torch.__version__, numpy=np.__version__, h5py=h5py.__version__)
        # Normalize tuples / paths before comparing with an existing serialized protocol.
        protocol = json.loads(json.dumps(protocol))
        if args.resume:
            if read_json(out / 'protocol.json') != protocol:
                raise ValueError('Resume settings, source data metadata or software differ from the recorded protocol')
        else:
            save_json(out / 'protocol.json', protocol)
        protocol_sha = digest(out / 'protocol.json')
        for fold in FOLDS:
            save_json(out / f'fold_{fold}_selection.json', data[fold]['summary'])
        save_json(out / 'environment.json', dict(executable=sys.executable, python=sys.version,
            torch=torch.__version__, numpy=np.__version__, h5py=h5py.__version__, device=args.device,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_runtime=torch.version.cuda, available_disk_gib=shutil.disk_usage(out).free / 1024**3))
        # Every invocation gets a fresh tiny view. No old or shared files are replaced.
        view = out / ('selected_data_view_' + stamp())
        for fold in FOLDS:
            create_scoring_view(view, fold, data[fold])
        print('[verify] Comparing selected content and metadata against all 156 supplied reference hashes', flush=True)
        verify_reference(out, view, data, support)
        if args.check_only:
            save_json(out / 'status.json', dict(status='CHECK_ONLY_COMPLETE', training_started=False))
            print(f'[checked] No training started. Audit: {out}', flush=True)
            package(out)
            return
        device = training_device(args)
        jobs = {}
        for fold in FOLDS:
            save_json(out / 'status.json', dict(status='TRAINING', fold=fold))
            jobs[fold] = run_fold(out, fold, data[fold], args, cfg, device, protocol_sha)
            package(out)
        evaluation = out / ('evaluation_' + stamp())
        evaluation.mkdir()
        seeds, all_rows = [], []
        for seed in args.seeds:
            save_json(out / 'status.json', dict(status='NESTED_EVALUATION', seed=seed))
            result, rows = evaluate_seed(evaluation, seed, jobs, data, evaluator, annotate, view, args.release)
            seeds.append(result)
            all_rows += rows
            package(out)
        summarize(evaluation, seeds, all_rows, jobs, args)
        save_json(out / 'completion.json', dict(status='COMPLETE', evaluation=str(evaluation),
            selected_validation_ids=list(VAL_IDS), folds=4, seeds=args.seeds, training_jobs=4 * len(args.seeds),
            protocol_sha256=protocol_sha, completed_utc=datetime.now(timezone.utc).isoformat()))
        save_json(out / 'status.json', dict(status='COMPLETE', evaluation=str(evaluation)))
        archive = package(out)
        print(f'\nCOMPLETE. Results: {evaluation / "RESULTS.md"}\nSend back: {archive}', flush=True)
    except BaseException:
        failure = traceback.format_exc()
        (out / ('FAILED_' + stamp() + '.txt')).write_text(failure, encoding='utf-8')
        print(failure, flush=True)
        print(f'Outputs retained in {out}. Completed jobs can be reused with --resume.', flush=True)
        try:
            package(out)
        except Exception:
            pass
        raise
    finally:
        sys.stdout = console
        log.close()


if __name__ == '__main__':
    main()
