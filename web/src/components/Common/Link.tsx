import type { HTMLProps, PropsWithChildren } from 'react'
import { useMemo } from 'react'

import type { StrictOmit } from 'ts-essentials'
import isAbsoluteUrl from 'is-absolute-url'
import NextLink, { LinkProps as NextLinkProps } from 'next/link'
import styled from 'styled-components'
import { GoLinkExternal } from 'react-icons/go'

export interface LinkInternalProps extends PropsWithChildren<NextLinkProps & HTMLProps<HTMLAnchorElement>> {
  className?: string
}

export function LinkInternal({ className, children, href, ...restProps }: LinkInternalProps) {
  return (
    <NextLink href={href} passHref={false}>
      {/* eslint-disable-next-line jsx-a11y/anchor-is-valid */}
      <a className={className}>{children}</a>
    </NextLink>
  )
}

const LinkExternalIconWrapper = styled.span<{ $color?: string }>`
  color: ${(props) => props.$color ?? '#777'};
`

export interface LinkExternalProps extends Omit<HTMLProps<HTMLAnchorElement>, 'as' | 'ref'> {
  href?: string
  $color?: string
  $iconColor?: string
  icon?: React.ReactNode
}

const A = styled.a<{ $color?: string } & LinkExternalProps>`
  color: ${(props) => props.$color ?? undefined};
  text-decoration: none;

  &:hover {
    color: ${(props) => props.$color ?? undefined};
    text-decoration: none;
  }

  white-space: nowrap;
`

export const ContentWrapper = styled.span`
  white-space: normal;
`

export function LinkExternal({
  href,
  $color,
  $iconColor,
  icon,
  children,
  ...restProps
}: PropsWithChildren<LinkExternalProps>) {
  const Icon: React.ReactNode = icon === undefined ? <GoLinkExternal /> : icon

  return (
    <>
      <A target="_blank" rel="noopener noreferrer" href={href} $color={$color} {...restProps}>
        <ContentWrapper>{children}</ContentWrapper>{' '}
        {Icon && <LinkExternalIconWrapper $color={$iconColor}>{Icon}</LinkExternalIconWrapper>}
      </A>
    </>
  )
}

export interface LinkProps extends StrictOmit<LinkInternalProps, 'href'> {
  href?: string
}

export function Link({ href, ...restProps }: LinkProps & LinkExternalProps) {
  const isExternal = useMemo(() => isAbsoluteUrl(href ?? ''), [href])

  if (!href) {
    return <span {...restProps} />
  }

  if (isExternal) {
    return <LinkExternal href={href} {...restProps} />
  }

  return <LinkInternal href={href} {...restProps} />
}
