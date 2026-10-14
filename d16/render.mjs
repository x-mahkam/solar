import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';
import { spawn } from 'child_process';
const [a, b, out] = [+process.argv[2], +process.argv[3], process.argv[4]];
const FPS = 30;
const br = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
const p = await br.newPage({ viewport:{width:1080,height:1920} });
p.setDefaultTimeout(600000);
await p.goto('file://'+process.cwd()+'/day.html?q=0.75&t=0');
await p.waitForFunction(()=>window.setTime);
const ff = spawn('ffmpeg', ['-v','error','-y','-f','image2pipe','-framerate',String(FPS),'-c:v','mjpeg','-i','-','-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p',out], { stdio:['pipe','inherit','inherit'] });
for (let f = a; f < b; f++) {
  await p.evaluate(t=>window.setTime(t), f / FPS);
  const buf = await p.screenshot({ type:'jpeg', quality:95 });
  if (!ff.stdin.write(buf)) await new Promise(r=>ff.stdin.once('drain', r));
  if (f % 60 === 0) console.log(out, f);
}
ff.stdin.end(); await new Promise(r=>ff.on('close', r)); await br.close(); console.log('DONE', out);
