import json
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).parent; OUT=D.parent/'assets';OUT.mkdir(exist_ok=True)
s=json.loads((D/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig,ax=plt.subplots(figsize=(9,4.8)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
labels=['Repeatable bonus\n(proxy reward)','One-time bonus + finish\n(corrected reward)']
vals=[100*s['proxy']['finish_rate'],100*s['corrected']['finish_rate']]
bars=ax.bar(labels, vals, width=.55, color=['#bc6c5c','#276e70'])
ax.set_ylim(0,115);ax.set_ylabel('Completed the race (%)');ax.set_title('The task did not change. Only the reward did.',fontsize=15,pad=18,fontweight='bold')
for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+3,f'{v:.0f}% of 20 trained agents',ha='center',fontweight='bold',fontsize=11)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(OUT/'reward_comparison.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

# Tile diagram for the two learned policies (real outcomes, not illustrative targets)
fig,ax=plt.subplots(figsize=(10,3.4));ax.set_xlim(-.5,6.5);ax.set_ylim(-.8,2.4);ax.axis('off')
for y,label,route in [(1.6,'Proxy: loops 3 → 4 → 3',s['proxy']['representative_seed0']['trace']),(0.3,'Corrected: reaches 6',s['corrected']['representative_seed0']['trace'])]:
 for i in range(7):
  bg='#f5ede5' if i==3 else ('#e7f1eb' if i==6 else '#f2f3f5')
  ax.add_patch(plt.Rectangle((i-.44,y-.31),.88,.62,facecolor=bg,edgecolor='#c7c9ce',linewidth=1))
  ax.text(i,y,str(i),ha='center',va='center',fontweight='bold')
 ax.text(-.47,y+.51,label,fontsize=11,fontweight='bold')
 ax.plot(route[:10], [y-.51]*min(10,len(route)),marker='o',markersize=4,alpha=.8)
ax.text(3,-.4,'tile 3 = repeatable reward in proxy setting',ha='center',fontsize=10)
fig.savefig(OUT/'learned_paths.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

# GIF actual trace, compact simulation visualization
p=s['proxy']['representative_seed0']['trace'];f=s['corrected']['representative_seed0']['trace']
W,H=1050,340
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21);title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',23)
except:font=ImageFont.load_default();title=font
frames=[]
for t in range(20):
 im=Image.new('RGB',(W,H),'white');dr=ImageDraw.Draw(im)
 dr.text((40,16),'The same seven tiles. Two different learned policies.',font=title,fill='#1b2533')
 for row,seq,name,col in [(0,p,'Proxy reward: chasing bonus','#b14d41'),(1,f,'Corrected reward: finishing','#23756e')]:
  y=95+row*133;dr.text((40,y-31),name,font=font,fill=col)
  for i in range(7):
   x=55+i*138
   dr.rounded_rectangle((x,y,x+111,y+67),radius=10,fill=('#fceccb' if i==3 else '#d7eee4' if i==6 else '#f1f3f6'),outline='#c6cad0',width=2)
   dr.text((x+47,y+20),str(i),fill='#2b3340',font=font)
  at=seq[min(t,len(seq)-1)];xx=55+138*at
  dr.ellipse((xx+40,y+38,xx+76,y+73),fill=col,outline='white',width=2)
  dr.text((665,y+77),'Step '+str(min(t,len(seq)-1)),font=font,fill='#555b63')
 frames.append(im)
frames[0].save(OUT/'reward_hacking_replay.gif',save_all=True,append_images=frames[1:],duration=290,loop=0,optimize=True)
print([ (f.name,f.stat().st_size) for f in OUT.iterdir() ])
