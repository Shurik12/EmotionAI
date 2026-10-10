from PIL import Image, ImageDraw, ImageFont
import math, json, base64, pathlib, zipfile
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=ROOT/'assets/ori_parts.png'
sheet=Image.open(SOURCE).convert('RGBA')
parts={}
for name,x0,x1 in [('head',0,540),('body',540,890),('left',890,1200),('right',1200,1478),('happy',1478,1983)]:
 im=sheet.crop((x0,0,x1,sheet.height)); box=im.getchannel('A').point(lambda x:255 if x>32 else 0).getbbox();im=im.crop(box)
 im.save(ROOT/'assets'/f'{name}.png');parts[name]=im
W,H=760,480
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n): return ImageFont.truetype(FONT,n)
CASES=[
(1,'Проверка состояния','check',900,'Ты сейчас в задаче?',['Да, продолжаю','Помоги вернуться','Нужна пауза']),
(2,'Предложить помощь','help',700,'Нужна помощь с текущим действием?',['Разбить на шаги','Продолжить самостоятельно','Сделать паузу']),
(3,'Поддержать','support',900,'Как тебе удобнее продолжить?',['Продолжить работу','Нужна помощь','Сделать паузу']),
(4,'Напомнить действие','point',800,'Напомнить текущее действие?',['Показать','Продолжить самостоятельно','Сделать паузу']),
(5,'Радость / интерес','still',0,'',[]),(6,'Удивление','still',0,'',[]),(7,'Стыд / презрение','still',0,'',[]),(8,'Обычная работа','still',0,'',[]),
(9,'Вернуться к действию','point',800,'Текущее действие — заполнить следующий пункт.',['Продолжить','Разбить на шаги','Сделать паузу']),
(10,'Продолжить без помощи','dismiss',150,'Ты сейчас в задаче?',['Да, продолжаю','Помоги вернуться','Нужна пауза']),
(11,'Запросить помощь','replace',0,'Что поможет продолжить?',['Показать текущее действие','Разбить на шаги','Сделать паузу']),
(12,'Разбить на шаги','steps',0,'Предлагаю такие шаги:',['Использовать эти шаги','Изменить','Назад']),
(13,'Пауза','pause',600,'Пауза. Можно вернуться в любой момент.',['Вернуться к задаче']),
(14,'Действие завершено','done',650,'Готово. Следующее действие — проверить результат.',['Продолжить']),
(15,'Нет ответа','timeout',200,'Ты сейчас в задаче?',['Да, продолжаю','Помоги вернуться','Нужна пауза']),
(16,'Анализ недоступен','unavailable',0,'Анализ временно недоступен.',[])]
def ease(x):x=max(0,min(1,x));return x*x*(3-2*x)
def pulse(t,a,b,c):return ease(t/a) if t<a else 1 if t<b else 1-ease((t-b)/(c-b))
def pose(kind,t):
 r=dict(head=0,lift=0,left=0,right=0,nod=0,happy=0)
 if kind=='check':
  p=pulse(t,450,650,900);r.update(head=6*p,lift=-6*p,left=-102*p)
 elif kind=='help':r['head']=5*pulse(t,250,450,700)
 elif kind=='support':
  p=pulse(t,300,500,900);r.update(left=62*p,right=-62*p)
 elif kind=='point':
  p=pulse(t,400,550,800);r.update(head=5*p,right=-72*p)
 elif kind=='pause':
  p=ease(t/600);r.update(nod=5*p,left=-8*p,right=8*p)
 elif kind=='done':
  r['nod']=7*pulse(t,200,200,450);r['happy']=ease((t-450)/200)
 return r
# Animation compositing: generated cutout parts rotate about shoulder/neck pivots.
def layer(canvas,name,xy,size,anchor=(0,0),angle=0,alpha=1):
 im=parts[name].resize(size,Image.Resampling.LANCZOS)
 if alpha<1:im.putalpha(im.getchannel('A').point(lambda a:int(a*alpha)))
 tile=Image.new('RGBA',(900,900));pivot=(450,450)
 tile.alpha_composite(im,(round(450-anchor[0]),round(450-anchor[1])))
 tile=tile.rotate(-angle,Image.Resampling.BICUBIC,center=pivot)
 canvas.alpha_composite(tile,(round(xy[0]-450),round(xy[1]-450)))
def wrap(txt,width,f):
 words=txt.split(); lines=[];s=''
 for word in words:
  test=(s+' '+word).strip()
  if f.getlength(test)>width and s:lines.append(s);s=word
  else:s=test
 if s:lines.append(s)
 return lines
def render(case,ms):
 num,title,kind,duration,msg,buttons=case;t=max(0,ms-500);p=pose(kind,t)
 out=Image.new('RGBA',(W,H),'#F5F5F1');d=ImageDraw.Draw(out)
 d.text((28,20),f'{num:02} · {title}',fill='#273334',font=font(23))
 d.text((28,54),'ОРИ / демонстрация поведения',fill='#596765',font=font(13))
 d.ellipse((115,406,259,421),fill='#E5E8E1')
 layer(out,'body',(190,402),(124,178),(62,178))
 layer(out,'left',(134,262),(82,108),(67,15),p['left'])
 layer(out,'right',(246,262),(82,108),(15,15),p['right'])
 hh=round(159*(1-p['nod']*.009));lift=p['lift']+p['nod']*.7
 layer(out,'head',(190,273+lift),(220,hh),(110,hh),p['head'])
 if p['happy']>0:layer(out,'happy',(190,273+lift),(220,hh),(110,hh),p['head'],p['happy'])
 opacity=1;display=msg;bs=buttons
 if kind=='still':opacity=0
 elif kind=='dismiss':opacity=1-ease(t/150)
 elif kind=='timeout':opacity=1-ease((ms-20000)/200)
 elif kind in ['replace','steps'] and ms<500:display='Текущее действие';bs=[]
 elif kind not in ['unavailable','replace','steps','timeout']:opacity=ease(t/150)
 if kind=='dismiss':opacity=1-ease(t/150)
 if opacity>0 and display:
  card=Image.new('RGBA',(W,H));cd=ImageDraw.Draw(card);x,y=358,145
  cd.rounded_rectangle((x,y,732,407),18,fill='#FFFFFF',outline='#D7E0DD',width=2)
  lines=wrap(display,330,font(19));yy=y+20
  for line in lines:cd.text((x+20,yy),line,fill='#273334',font=font(19));yy+=26
  if kind=='steps' and ms>=500:
   for line in ['1. Открыть нужный документ','2. Заполнить следующий пункт','3. Проверить результат']:
    cd.text((x+20,yy+4),line,fill='#596765',font=font(13));yy+=21
  for i,b in enumerate(bs):
   cd.rounded_rectangle((x+18,yy+8,x+354,yy+39),8,fill='#426B64' if i==0 else '#E8EEF4')
   cd.text((x+30,yy+15),b,fill='white' if i==0 else '#273334',font=font(13));yy+=40
  card.putalpha(card.getchannel('A').point(lambda a:round(a*opacity)));out.alpha_composite(card)
 d=ImageDraw.Draw(out)
 if kind=='still':label='Без движения и звука'
 elif kind=='timeout':label='20 секунд ожидания → исчезновение карточки за 200 мс'
 elif kind in ['replace','steps','dismiss','unavailable']:label='Ори неподвижен · меняется только интерфейс'
 else:label=f'Один жест · {duration} мс · затем покой'
 d.text((28,449),label,fill='#596765',font=font(13))
 return out.convert('RGB')
manifest=[]
for c in CASES:
 n,title,kind,ms,_,_=c
 if kind=='timeout':times=[0]+list(range(20000,20201,50))+[20250];dur=[20000]+[50]*5+[1300]
 elif kind=='still' or kind=='unavailable':times=[0];dur=[2000]
 else:
  total=500+max(ms,600)+1200;times=list(range(0,total,50));dur=[50]*len(times)
 palette=render(c,950).quantize(colors=256)
 frames=[render(c,t).quantize(palette=palette,dither=Image.Dither.NONE) for t in times]
 path=ROOT/'gifs'/f'{n:02}_{kind}.gif'
 frames[0].save(path,save_all=True,append_images=frames[1:],duration=dur,loop=0,optimize=True,disposal=2)
 manifest.append(dict(id=n,title=title,kind=kind,duration_ms=ms,file='gifs/'+path.name,loop='preview only; production play once',sound=n in [1,14]))
 print(path.name,flush=True)
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
# Contact sheet of representative motion frames for QA.
contact=Image.new('RGB',(1520,1440),'white')
for j,n in enumerate([1,2,3,4,13,14]):contact.paste(render(CASES[n-1],950),(j%2*760,j//2*480))
contact.save(ROOT/'contact.png')
# Single-file offline viewer; embeds all animated GIFs, supports restart and optional synthesized sound.
encoded=[]
for c,m in zip(CASES,manifest):encoded.append(dict(**m,data='data:image/gif;base64,'+base64.b64encode((ROOT/m['file']).read_bytes()).decode()))
html='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ори — 16 сценариев</title><style>body{margin:0;background:#F5F5F1;color:#273334;font:16px system-ui}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:36px;margin:0 0 12px}p{line-height:1.6;color:#596765}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}article{background:white;border:1px solid #d7e0dd;border-radius:18px;overflow:hidden}img{width:100%;display:block}button{border:0;border-radius:8px;background:#426B64;color:white;padding:10px 16px;cursor:pointer;margin:12px}label{display:block;margin:20px 0}small{color:#596765}footer{padding:24px 0}</style><main><h1>Ори · 16 сценариев</h1><p>Рабочие анимационные прототипы по таблице. Жесты проигрываются один раз внутри каждого цикла GIF; повторение здесь нужно для просмотра. В приложении — только один запуск на событие. В статичных сценариях персонаж не двигается.</p><label><input id="sound" type="checkbox"> Звук при нажатии «Повторить»: только сценарии 1 и 14</label><div id="grid"></div><footer>Сценарий 15: карточка действительно ждёт 20 секунд. GIF не содержит аудио. Звук синтезируется только после нажатия кнопки и включения настройки. Карточки и кнопки внутри GIF — демонстрационные.</footer></main><script>const cases=DATA;let ctx;function tone(f,t,d,a,r){const o=ctx.createOscillator(),g=ctx.createGain();o.type='sine';o.frequency.value=f;g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(.0631,t+a);g.gain.setValueAtTime(.0631,t+d-r);g.gain.linearRampToValueAtTime(0,t+d);o.connect(g);g.connect(ctx.destination);o.start(t);o.stop(t+d)}function play(n){if(!document.querySelector('#sound').checked)return;ctx ||=new AudioContext();ctx.resume().then(()=>{const t=ctx.currentTime+.5;if(n===1)tone(440,t,.4,.08,.2);if(n===14){tone(523,t,.16,.03,.08);tone(659,t+.22,.22,.03,.08)}})}cases.forEach(c=>{const a=document.createElement('article'),im=new Image();im.src=c.data;im.alt=c.title;a.append(im);const b=document.createElement('button');b.textContent='Повторить';b.onclick=()=>{const next=new Image();next.src=c.data;next.alt=c.title;a.replaceChild(next,a.querySelector('img'));play(c.id)};a.append(b);const dl=document.createElement('a');dl.href=c.data;dl.download=c.file.split('/').pop();dl.textContent='Скачать GIF';dl.style='margin-right:12px;color:#426B64';a.append(dl);const s=document.createElement('small');s.textContent=c.sound?'Сигнал доступен по кнопке':'Без звука';a.append(s);document.querySelector('#grid').append(a)});</script></html>'''.replace('DATA',json.dumps(encoded,ensure_ascii=False))
(ROOT/'ORI_16_Animations.html').write_text(html)
(ROOT/'README.txt').write_text('ОРИ — 16 анимационных прототипов\n\nОткройте ORI_16_Animations.html в браузере для просмотра всех сценариев.\nGIF-файлы находятся в gifs/. Звук доступен в HTML после включения настройки и нажатия «Повторить».\nGIF циклически повторяются только для демонстрации; в приложении запускать жест один раз на событие.\nСтатичные сценарии 5–8 и 16 намеренно не содержат движения. Сценарий 15 ждёт 20 секунд.\nКарточки внутри GIF демонстрационные. Исходные слои персонажа — assets/.\nЭто анимация плоских слоёв (2D cutout), а не готовая 3D-модель. Наклон головы заменяет объёмный поворот; точную объёмную артикуляцию можно добавить после 3D-риггинга.\nВсе эмоциональные правила остаются экспериментальными: анимация не подтверждает точность распознавания состояния.\n',encoding='utf-8')
