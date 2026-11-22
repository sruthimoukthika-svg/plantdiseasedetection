"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/scanner", label: "Disease Scanner" },
  { href: "/plants", label: "Supported Plants" },
  { href: "/about", label: "About the AI" },
];

export default function Nav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-40 border-b border-leaf-100 bg-canvas/85 backdrop-blur">
      <nav
        aria-label="Main"
        className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6"
      >
        <Link href="/" className="flex items-center gap-2.5">
          <span
            aria-hidden
            className="grid h-8 w-8 place-items-center rounded-lg bg-leaf-600 text-surface"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor">
              <path d="M12 21c-.4-4.2.6-7.3 2.2-9.4 1.7-2.2 4-3.4 6.3-3.9-.2 3.4-1.3 6-3.1 7.7-1.8 1.7-4 2.5-5.4 2.7V21z" />
              <path d="M11 21c.3-3.1-.4-5.5-1.6-7.2C8.1 12 6.3 11.1 4.5 10.7c.2 2.6 1 4.6 2.4 5.9 1.3 1.3 3 1.9 4.1 2.1V21z" opacity=".55" />
            </svg>
          </span>
          <span className="text-[15px] font-semibold tracking-tight">
            PlantHealth<span className="text-leaf-600">AI</span>
          </span>
        </Link>

        <ul className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => {
            const active = path === l.href;
            return (
              <li key={l.href}>
                <Link
                  href={l.href}
                  aria-current={active ? "page" : undefined}
                  className={`rounded-lg px-3 py-2 text-sm transition-colors ${
                    active
                      ? "bg-leaf-100 font-medium text-leaf-800"
                      : "text-ink-muted hover:bg-leaf-50 hover:text-leaf-700"
                  }`}
                >
                  {l.label}
                </Link>
              </li>
            );
          })}
          <li className="ml-2">
            <Link
              href="/scanner"
              className="rounded-lg bg-leaf-600 px-3.5 py-2 text-sm font-medium text-surface transition-colors hover:bg-leaf-700"
            >
              Scan a Plant
            </Link>
          </li>
        </ul>

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls="mobile-menu"
          aria-label="Toggle navigation"
          className="rounded-lg p-2 text-ink-muted hover:bg-leaf-50 md:hidden"
        >
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
            {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </nav>

      {open && (
        <ul id="mobile-menu" className="border-t border-leaf-100 bg-surface px-4 py-2 md:hidden">
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link
                href={l.href}
                onClick={() => setOpen(false)}
                className={`block rounded-lg px-3 py-2.5 text-sm ${
                  path === l.href
                    ? "bg-leaf-100 font-medium text-leaf-800"
                    : "text-ink-muted"
                }`}
              >
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </header>
  );
}
