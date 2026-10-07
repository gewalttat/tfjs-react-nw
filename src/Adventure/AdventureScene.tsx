import React from 'react';

/** Small integer coordinates keep the illustration crisp at every viewport size. */
export function AdventureScene() {
  return <svg className="adventure-landscape" viewBox="0 0 960 220" preserveAspectRatio="xMidYMax slice" aria-hidden="true" shapeRendering="crispEdges">
    <defs>
      <linearGradient id="night" x2="0" y2="1"><stop stopColor="#071019" /><stop offset="1" stopColor="#26353d" /></linearGradient>
      <radialGradient id="moon-halo"><stop stopColor="#e8d59a" stopOpacity=".19" /><stop offset="1" stopColor="#d5dcb5" stopOpacity="0" /></radialGradient>
      <radialGradient id="window-halo"><stop stopColor="#ffb74d" stopOpacity=".35" /><stop offset="1" stopColor="#ed9b38" stopOpacity="0" /></radialGradient>
      <linearGradient id="stone"><stop stopColor="#3f4b53" /><stop offset=".45" stopColor="#28353e" /><stop offset="1" stopColor="#101c25" /></linearGradient>
      <linearGradient id="plaster"><stop stopColor="#4d4537" /><stop offset="1" stopColor="#292925" /></linearGradient>
      <pattern id="masonry" width="16" height="12" patternUnits="userSpaceOnUse"><path d="M0 0h16M0 6h16M8 0v6M0 6v6" fill="none" stroke="#0a171e" strokeOpacity=".55" /><path d="M1 1h6m2 6h6" stroke="#78858a" strokeOpacity=".2" /></pattern>
      <pattern id="shingles" width="12" height="8" patternUnits="userSpaceOnUse"><path fill="#25343c" d="M0 0h12v8H0z" /><path d="M0 7h12M6 0v7" stroke="#09151c" /><path d="M1 1h4" stroke="#4f6064" /></pattern>
    </defs>
    <g className="parallax-sky">
    <path fill="url(#night)" d="M0 0h960v220H0z" />
    {Array.from({ length: 45 }, (_, i) => <rect className="scene-star" key={i} x={(i * 137 + 29) % 960} y={(i * 41 + 7) % 110} width={i % 4 === 0 ? 2 : 1} height={i % 4 === 0 ? 2 : 1} fill="#c3c7b4" style={{ animationDelay: `${i % 7}s` }} />)}
    <ellipse cx="811" cy="48" rx="105" ry="90" fill="url(#moon-halo)" /><g className="scene-moon"><path fill="#e9bc64" d="M799 24h24v4h8v8h4v24h-4v8h-8v4h-24v-4h-8v-8h-4V36h4v-8h8z" /><path fill="#c39246" d="M798 34h8v8h-8zm19 12h10v12h-10zm-17 13h7v5h-7z" /><path fill="#ffe2a1" d="M798 29h18v3h-15v7h-5v-5h2z" /></g>
    <g className="scene-clouds" fill="#74818f" opacity=".08"><path d="M540 42h60v-6h40v4h61v7h46v6H522v-5h18zM193 70h42v-6h58v5h83v7H171v-3h22z" /></g>
    <path className="shooting-star" stroke="#e5e5c5" strokeWidth="1" d="M595 15l-30 13" />
    </g><g className="parallax-distance">
    {/* Distant ridges have stepped silhouettes and moonlit rock faces. */}
    <path fill="#26343e" d="M0 139h19v-12h16v-16h15V96h10V84h9v12h12v16h18v18h14v-22h12V91h10V71h10V57h9v17h12v19h14v21h15v23h17v-17h17v-16h13V85h10v17h15v23h21v25h22v-25h16v-23h14V83h10V67h8V47h9v22h12v19h17v23h18v20h20v22h25v-19h17v-16h16V99h12v16h17v20h24v-25h15V90h12V70h10V49h9v20h13v22h17v21h23v22h26v-23h15V92h13V75h9v16h15v25h22v25h25v-18h13v-23h14V76h12V55h8V37h8v22h12v22h18v18h20v22h25v28h28v71H0z" />
    <path fill="#465159" opacity=".55" d="M145 57h9v17h12v19h14v21h15v23h-9v-15h-13v-20h-12V88h-9V74h-7zm215-10h9v22h12v19h17v23h-12V99h-10V80h-9V67h-7zm412-10h8v22h12v22h18v18h-9V88h-15V71h-8V54h-6z" />
    {Array.from({ length: 110 }, (_, i) => <rect key={`rock-${i}`} x={(i * 83 + 7) % 960} y={112 + (i * 13) % 66} width={3 + i % 6} height="2" fill={i % 2 ? '#354650' : '#192a34'} opacity=".5" />)}
    <path fill="#16262f" d="M0 166h35v-9h34v-12h24v9h32v20h42v-11h25v-16h19v12h33v16h45v-9h39v-16h25v12h35v14h38v-11h37v-13h21v10h40v19h46v-13h30v-20h27v12h30v20h42v-13h27v-19h25v11h26v16h35v-13h35v-10h29v16h38v-8h28v-15h25v30h68v44H0z" />
    </g><g className="parallax-world">
    {/* Three tree layers, irregular branches and cool rim lighting. */}
    {[0, 1, 2].map((layer) => <g key={layer}>{Array.from({ length: 18 }, (_, i) => {
      const x = (i * 71 + layer * 37) % 960, y = 126 + layer * 27 + i % 3 * 8;
      const scale = .7 + layer * .16 + i % 4 * .08;
      return <g key={i} transform={`translate(${x} ${y}) scale(${scale})`}>
        <path fill={['#14272f', '#0e2027', '#09191e'][layer]} d="M-2-57h4v8h4v8h5v7h-4v4h9v7h-4v5h12v7h-6v4h15v8h-8v4h12v8h-26v28h-5V23h-25v-8h10v-4h-7V3h13v-4h-5v-7h10v-5h-4v-7h8v-4h-3v-7h4v-9h3z" />
        <path fill={layer === 2 ? '#23373a' : '#32464c'} opacity=".6" d="M-2-50h2v12h-3v7h-4v3h6v3h-9v5h-6v3h8v3h-13v5h-6v3h11v3h-15v5h-5v3h10v3h-14v-3h-3V3h12v-4h-5v-7h10v-5h-4v-7h8v-4h-3v-7h4v-9h3z" />
        <path stroke="#3a3e32" strokeWidth="2" d="M0 5v38" /><path stroke="#0a171b" d="M2 9v34" />
      </g>;
    })}</g>)}
    {/* The castle: roof tiles, buttresses, masonry and recessed windows. */}
    <path fill="#0a141c" d="M775 213v-8h12v-9h12v-6h112v7h19v9h16v14H771z" />
    <g fill="url(#stone)" stroke="#0b151e" strokeWidth="2">
      <path d="M824 126h76v79h-76zm-20-43h27v122h-27zm87 10h30v112h-30zM842 112h42v93h-42z" />
    </g>
    <g fill="url(#masonry)"><path d="M824 126h76v79h-76zm-20-43h27v122h-27zm87 10h30v112h-30zM842 112h42v93h-42z" /></g>
    <g fill="url(#shingles)" stroke="#09131c" strokeWidth="2"><path d="M797 84h41l-4-7h-4v-8h-4v-9h-4v-10h-5v-8h-4v12h-4v12h-4v10h-4zM885 94h43l-4-8h-4V75h-5V62h-5V49h-5v13h-5v13h-5v10h-5zM837 113h51l-6-7h-6v-7h-6v-8h-7v-7h-6v8h-6v7h-7v7h-7z" /></g>
    <path fill="#627076" opacity=".5" d="M804 85h3v120h-3zm87 11h3v109h-3zm-49 19h3v90h-3z" />
    <path fill="#0b1820" d="M825 87h5v118h-5zm89 10h6v108h-6zm-37 18h6v90h-6z" />
    <path fill="#3b4b50" d="M799 83h37v4h-37zm88 11h39v4h-39zm-51 19h53v4h-53zm-11 33h14v4h-14zm60 0h9v4h-9z" />
    {[[811, 101], [811, 136], [899, 119], [899, 156], [850, 132], [873, 158]].map(([x, y], i) => <g key={i}>
      <ellipse cx={x + 3} cy={y + 6} rx="15" ry="24" fill="url(#window-halo)" />
      <path fill="#09131a" d={`M${x - 2} ${y + 13}v-13h2v-3h5v3h2v13z`} />
      <rect className="castle-window" x={x} y={y} width="5" height="11" fill="#ffc769" style={{ animationDelay: `${i * .7}s` }} />
      <path stroke="#714a2d" d={`M${x + 2} ${y}v11m-2-6h5`} /><path stroke="#59605b" d={`M${x - 2} ${y + 13}h10`} />
    </g>)}
    <path fill="#101920" d="M854 205v-21h3v-5h10v5h3v21z" /><path fill="#3d3229" d="M857 205v-20h10v20z" /><path stroke="#8b7150" d="M862 185v20" />
    <path fill="#536052" d="M788 205h147v3H788zm-11 8h161v2H777z" />
    <g className="castle-enchantment" fill="none" stroke="#8bbaca" strokeWidth=".7" opacity=".6"><path d="M864 170l5 5-5 5-5-5zM864 166v4m0 10v4m-9-9h4m10 0h4" /></g>
    <g className="castle-banner"><path fill="#733e4d" d="M836 88h12v20l-6-5-6 5z" /><path stroke="#b09465" d="M835 87h15" /><path fill="#d7b574" d="M841 92h2v7h-2zm-2 2h6v2h-6z" /></g>
    <path fill="#091317" d="M0 200h134v4h53v4h580v-4h193v16H0z" />
    {/* Timber cottage with a tiled roof, chimney, shutters and warm spill light. */}
    <path fill="url(#plaster)" stroke="#091217" strokeWidth="2" d="M0 98h111v104H0z" />
    {Array.from({ length: 70 }, (_, i) => <rect key={`wall-${i}`} x={(i * 37) % 110} y={109 + (i * 17) % 87} width={2 + i % 4} height="2" fill={i % 3 ? '#695643' : '#1a2626'} opacity=".45" />)}
    <path fill="#29383b" d="M0 77h96l29 26H0z" />
    {Array.from({ length: 5 }, (_, row) => <g key={row}>{Array.from({ length: 12 }, (_, col) => <path key={col} fill={(col + row) % 3 === 0 ? '#4c5145' : '#323f40'} stroke="#17282d" strokeWidth="1" d={`M${col * 10 - row % 2 * 5} ${78 + row * 5}h8l5 5h-10z`} />)}</g>)}
    <path fill="#19282e" d="M83 60h15v31H83z" /><path fill="#45504e" d="M81 59h19v5H81z" /><path fill="#172128" d="M86 57h11v3H86z" />
    <g className="chimney-smoke" fill="#8b9a99" opacity=".16"><path d="M88 54h8v-7h-5v-6h-9v-5h-6v-7h-7v8h4v8h8v5h7z" /></g>
    <path stroke="#8b6841" strokeWidth="4" d="M0 106h113M17 108v90m65-90v90M0 190h115" /><path stroke="#271f1b" strokeWidth="2" d="M20 109v81m48-80L21 140m46 0l-47 34" />
    <ellipse cx="46" cy="151" rx="52" ry="56" fill="url(#window-halo)" />
    <path fill="#182025" stroke="#a0804d" strokeWidth="2" d="M28 122h36v58H28z" />
    <path className="castle-window" fill="#eac075" d="M33 128h26v47H33z" /><path fill="#fff0b3" opacity=".55" d="M35 130h8v41h-8z" />
    <path stroke="#52402c" strokeWidth="3" d="M46 127v49m-14-25h28" />
    <path fill="#344036" stroke="#111c20" d="M22 127h6v46h-6zm42 0h8v46h-8z" /><path stroke="#60705a" d="M23 131v37m44-37v37" />
    <path fill="#8e7045" d="M26 178h41v4H26z" /><path fill="#61482e" d="M91 140h18v50H91z" /><path fill="#24251e" d="M94 143h12v44H94z" /><rect fill="#c39b55" x="102" y="162" width="2" height="3" />
    <path fill="#b09161" d="M85 186h34v4H85zm-7 6h32v4H78zm10 6h38v4H88z" /><path fill="#382c23" d="M85 190h34v2H85zm-7 6h32v2H78z" />
    <path fill="#304238" d="M5 196h8v-8h3v12h5v-5h3v7H5zm63 5v-9h3v-4h3v13h5v-7h3v10H68z" />
    <path fill="#6b563a" d="M4 203h89v2H4zm111 4h43v2h-43z" />
    </g>
    <g className="parallax-foreground">
      {/* A silver stream, luminous mushrooms and an owl on the cottage roof. */}
      <path fill="#18363d" d="M700 201h32l-18 5h24l-14 5h24l25 9H667l38-9h-18l25-5h-25z" />
      <g className="water-shimmer" fill="#8aaea9" opacity=".4"><path d="M704 204h17v1h-17zm-3 5h26v1h-26zm-13 6h20v1h-20zm32 1h24v1h-24z" /></g>
      {[[144, 207], [164, 213], [752, 209], [762, 212]].map(([x, y], i) => <g key={i} className="glow-mushroom" style={{ animationDelay: `${i}s` }}><path fill="#698e86" d={`M${x} ${y}h2v6h-2z`} /><path fill="#87b8b7" d={`M${x - 3} ${y}v-3h2v-2h4v2h2v3z`} /><rect x={x - 1} y={y - 3} width="2" height="1" fill="#d1f2d8" /></g>)}
      <g transform="translate(108 87)"><path fill="#594f41" d="M-5 0v-10h3v3h4v-3h3V0H3v4h-6V0z" /><path fill="#ada080" d="M-3-6h6v5h-6z" /><g className="owl-eyes" fill="#efd07c"><rect x="-3" y="-5" width="2" height="2" /><rect x="1" y="-5" width="2" height="2" /></g><path fill="#c18b4c" d="M-1-2h2v2h-2z" /></g>
      <g fill="#111f20"><path d="M0 220v-13h4v7h4v-17h3v15h4v-10h3v18zm934 0v-13h4v7h4v-20h3v13h4v-9h4v22z" /></g>
    <g className="scene-fireflies" fill="#f4cb71"><rect x="733" y="173" width="2" height="2" /><rect x="285" y="188" width="2" height="2" /><rect x="676" y="158" width="2" height="2" /></g>
    </g>
  </svg>;
}

export function Wizard({ talking }: { talking: boolean }) {
  return <svg className={`wizard ${talking ? 'wizard-talking' : ''}`} viewBox="0 -12 142 128" role="img" aria-label="Wizard" shapeRendering="crispEdges">
    <defs><radialGradient id="campfire-light"><stop stopColor="#ffb74a" stopOpacity=".4" /><stop offset="1" stopColor="#ff8b32" stopOpacity="0" /></radialGradient></defs>
    <ellipse className="campfire-glow" cx="109" cy="101" rx="40" ry="13" fill="url(#campfire-light)" />
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
    {/* Cloth seams, embroidered cuff, face wrinkles and separate beard strands. */}
    <path fill="#202f33" d="M20 74h4v21h-4zm10 11h4v12h-4zm29-13h4v17h-4zm-17 20h4v7h-4z" />
    <path fill="#a89358" d="M24 79h6v2h-6zm37 6h8v2h-8z" />
    <path fill="#9b704c" d="M33 56h5v1h-5zm20 1h6v1h-6zm-17 4h4v1h-4zm19-1h4v1h-4z" />
    <path fill="#a7997b" d="M31 69h2v11h-2zm7 6h2v13h-2zm12 4h2v15h-2zm8-10h2v12h-2z" />
    <path fill="#ffe9b7" opacity=".45" d="M59 50h2v10h-2zm4 20h2v9h-2zm5 17h2v8h-2z" />
    {/* A curved wooden pipe, glowing tobacco and drifting puffs. */}
    <path fill="#291a15" d="M48 67h17v-3h10v-5h10v13h-4v4h-9v-5H61v-2H48z" />
    <path fill="#a16b3d" d="M49 67h14v-2h11v-5h9v10h-4v3h-5v-5H64v1H49z" />
    <path fill="#d39a52" d="M76 62h5v2h-5z" /><path className="pipe-ember" fill="#ff9850" d="M75 59h8v2h-8z" />
    {[0, 1, 2].map((i) => <g className="pipe-smoke" key={i} style={{ animationDelay: `${i * 1.2}s` }} fill="#b7c7c4"><path d="M77 56h5v-5h-3v-4h-5v-4h-4v-5h-4v6h3v5h5v4h3z" /></g>)}
    {/* Stone ring, charred logs, layered flames and sparks. */}
    <g fill="#52616a"><path d="M99 104h7v5h-7zm10 3h10v5h-10zm14-2h8v5h-8zm-30-5h6v5h-6zm37-2h6v6h-6z" /></g>
    <path fill="#392a20" d="M100 99l25 7 2-5-25-7zM103 106l25-10-3-4-25 10z" /><path stroke="#93653c" strokeWidth="1" d="M103 96l21 6m-20 1l21-9" />
    <g className="campfire-flames">
      <path fill="#bd4926" d="M103 101h24v-7h4v-8h-5v-9h-4v-9h-4v12h-5v-7h-3v12h-6v-5h-3v12h2z" />
      <path fill="#f38e32" d="M106 100h19V88h-4V77h-3v10h-5v-5h-3v11h-4z" />
      <path fill="#ffd368" d="M111 100h11v-8h-4v-7h-3v9h-4z" /><path fill="#fff0b0" d="M114 101h6v-6h-3v-4h-2v7h-1z" />
    </g>
    {[0, 1, 2, 3].map((i) => <rect className="campfire-spark" key={i} x={108 + i * 5} y={80 - i * 4} width="2" height="2" fill="#ffc965" style={{ animationDelay: `${i * .6}s` }} />)}
    <path fill="#172023" d="M18 97h19v7H15v-4h3zm35 0h17v7H51v-4h2z" />
  </svg>;
}
