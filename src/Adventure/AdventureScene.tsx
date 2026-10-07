import React from 'react';

/** Small integer coordinates keep the illustration crisp at every viewport size. */
export function AdventureScene() {
  return <svg className="adventure-landscape" viewBox="0 0 960 220" preserveAspectRatio="xMidYMax slice" aria-hidden="true" shapeRendering="crispEdges">
    <defs><linearGradient id="night" x2="0" y2="1"><stop stopColor="#081119" /><stop offset="1" stopColor="#18242b" /></linearGradient></defs>
    <path fill="url(#night)" d="M0 0h960v220H0z" />
    {Array.from({ length: 45 }, (_, i) => <rect className="scene-star" key={i} x={(i * 137 + 29) % 960} y={(i * 41 + 7) % 110} width={i % 4 === 0 ? 2 : 1} height={i % 4 === 0 ? 2 : 1} fill="#c3c7b4" style={{ animationDelay: `${i % 7}s` }} />)}
    <g className="scene-moon"><path fill="#e9bc64" d="M799 24h24v4h8v8h4v24h-4v8h-8v4h-24v-4h-8v-8h-4V36h4v-8h8z" /><path fill="#c39246" d="M798 34h8v8h-8zm19 12h10v12h-10zm-17 13h7v5h-7z" /><path fill="#ffe2a1" d="M798 29h18v3h-15v7h-5v-5h2z" /></g>
    <path fill="#26313a" d="M0 155l62-63 19 26 42-64 48 71 30-42 66 91 84-129 35 55 27-19 73 99 47-53 29 25 81-112 22 51 35-13 49 71 49-48 42 29 68-99 35 56 31-20 51 61v120H0z" />
    <path fill="#151f28" d="M0 174l67-41 44 26 64-69 51 76 48-34 67 43 44-57 32 41 55-20 52 40 69-49 45 37 83-77 61 77 47-20 67 30 59-56v99H0z" />
    {Array.from({ length: 28 }, (_, i) => { const x = (i * 79) % 960; const y = 128 + i % 4 * 12; return <g key={i} fill={i % 2 ? '#0c171c' : '#101c22'}><path d={`M${x} ${y - 45}l-8 15h4l-13 19h6l-17 23h56l-17-23h6l-13-19h4z`} /><rect x={x - 2} y={y + 7} width="4" height="30" /></g>; })}
    <g fill="#263039" stroke="#0b1319" strokeWidth="3">
      <path d="M824 127h76v80h-76zm-20-43h27v123h-27zm87 10h30v113h-30z" /><path fill="#0a131b" d="M797 84l20-42 21 42zm88 10l21-46 22 46z" />
      <path d="M842 113h42v94h-42z" /><path fill="#101820" d="M837 113l26-29 25 29z" />
    </g>
    {[ [811, 101], [811, 136], [899, 119], [899, 156], [850, 132], [873, 158] ].map(([x, y], i) => <rect key={i} className="castle-window" x={x} y={y} width="5" height="11" fill="#ffc769" style={{ animationDelay: `${i * .7}s` }} />)}
    <path fill="#060e12" d="M0 202h960v18H0z" /><path fill="#354036" d="M0 203h140v3H0zm740 0h220v3H740z" />
    <g stroke="#0a1014" strokeWidth="3"><path fill="#2d2925" d="M0 102h107v100H0z" /><path fill="#242d2f" d="M0 78h97l29 25H0z" /><path stroke="#705139" d="M0 104h111M17 105v97m65-97v97" /><rect fill="#181c1e" x="29" y="124" width="34" height="56" /><rect className="castle-window" fill="#eebd62" x="35" y="131" width="22" height="43" /><path stroke="#483c2d" d="M46 130v45m-13-22h25" /><path stroke="#705139" d="M0 190h119m-4-81v93" /></g>
    <g fill="#bd955b"><path d="M85 179h30v5H85zm-7 6h30v5H78zm12 6h35v5H90z" /></g>
    <g className="scene-fireflies" fill="#f4cb71"><rect x="733" y="173" width="2" height="2" /><rect x="285" y="188" width="2" height="2" /><rect x="676" y="158" width="2" height="2" /></g>
  </svg>;
}

export function Wizard({ talking }: { talking: boolean }) {
  return <svg className={`wizard ${talking ? 'wizard-talking' : ''}`} viewBox="0 0 96 112" role="img" aria-label="Wizard" shapeRendering="crispEdges">
    <path fill="#071014" d="M19 40h52v11h8v49H12V60h7z" />
    <path fill="#374244" d="M23 65h41l11 33H15z" /><path fill="#626c64" d="M23 68h7v23h-7zm36 4h6v22h-6z" />
    <path fill="#ddb378" d="M31 43h30v27H31z" /><path fill="#b48b58" d="M31 43h6v20h-6zm25 2h5v18h-5z" />
    <g className="wizard-eyes" fill="#162022"><path d="M37 51h4v4h-4zm14 0h4v4h-4z" /></g>
    <path fill="#f1e2be" d="M32 48h11v3H32zm17 0h12v3H49z" /><path fill="#bf8b56" d="M44 53h5v9h-5z" />
    <path fill="#d5c9a9" d="M29 61h9v5h17v-5h9v18h-6v10h-6v8H41v-6h-7V80h-5z" /><path fill="#f0e5c7" d="M34 64h5v16h-5zm10 6h5v21h-5zm11-6h4v17h-4z" />
    <path className="wizard-mouth" fill="#665042" d="M42 66h10v3H42z" />
    <path fill="#623728" d="M10 39h8v-8h7V20h10V9h21v9h7v13h10v8h8v9H10z" /><path fill="#9c5233" d="M19 35h9V23h9V13h16v9h7v15h11v6H19z" /><path fill="#e6be7e" d="M37 18h10v6h6v9H40v-5h-7v-7h4zM24 36h5v4h-5zm37 2h6v4h-6z" /><path fill="#3d2821" d="M12 43h67v6H12z" />
    <path fill="#866337" d="M78 34h4v72h-4zM78 28h12v4H78zm9 3h4v10h-4z" /><path fill="#e3b874" d="M69 76h13v9H69z" />
    <g className="wizard-lantern"><path fill="#332b20" d="M84 47h10v19H82V51h2z" /><path fill="#f6c66f" d="M85 51h6v11h-6z" /><path fill="#fff0ad" d="M87 54h2v5h-2z" /></g>
    <path fill="#172023" d="M18 97h19v7H15v-4h3zm35 0h17v7H51v-4h2z" />
  </svg>;
}
