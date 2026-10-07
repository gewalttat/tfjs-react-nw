import { useEffect, useRef } from 'react';

export function useParallax() {
  const element = useRef<HTMLElement>(null);
  useEffect(() => {
    const node = element.current;
    if (!node) return;
    const media = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    let frame = 0, x = 0, y = 0;
    const update = () => {
      frame = 0;
      if (media?.matches) { node.style.setProperty('--scene-x', '0px'); node.style.setProperty('--scene-y', '0px'); return; }
      const bounds = node.getBoundingClientRect();
      node.style.setProperty('--scene-x', `${x * 10}px`);
      node.style.setProperty('--scene-y', `${y * 5 + Math.max(-12, Math.min(12, -bounds.top * .035))}px`);
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    const move = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse') return;
      const bounds = node.getBoundingClientRect();
      x = ((event.clientX - bounds.left) / bounds.width - .5) * 2;
      y = ((event.clientY - bounds.top) / bounds.height - .5) * 2;
      schedule();
    };
    const leave = () => { x = 0; y = 0; schedule(); };
    node.addEventListener('pointermove', move); node.addEventListener('pointerleave', leave);
    window.addEventListener('scroll', schedule, { passive: true });
    media?.addEventListener?.('change', schedule);
    return () => {
      cancelAnimationFrame(frame); node.removeEventListener('pointermove', move); node.removeEventListener('pointerleave', leave);
      window.removeEventListener('scroll', schedule); media?.removeEventListener?.('change', schedule);
    };
  }, []);
  return element;
}
