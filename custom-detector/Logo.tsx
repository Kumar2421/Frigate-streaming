// @ts-nocheck - This file is a copy for Docker build; path aliases work in web directory context
import { cn } from "@/lib/utils";

type LogoProps = {
  className?: string;
};
export default function Logo({ className }: LogoProps) {
  return (
    <img
      src="/images/logo.png"
      alt="Frigate Logo"
      className={cn("object-contain", className)}
    />
  );
}
