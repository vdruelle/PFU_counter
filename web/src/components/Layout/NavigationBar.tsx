import React, { useMemo, useState } from 'react'

import { useRouter } from 'next/router'
import styled from 'styled-components'
import {
  Collapse,
  Nav as NavBase,
  Navbar as NavbarBase,
  NavbarToggler as NavbarTogglerBase,
  NavItem as NavItemBase,
  NavLink as NavLinkBase,
} from 'reactstrap'
import classNames from 'classnames'

import { Link } from 'src/components/Common/Link'

const navLinksLeft: Record<string, string> = {
  '/': 'Home',
}

export const Navbar = styled(NavbarBase)`
  box-shadow: none;
  background-color: transparent !important;
`

export const Nav = styled(NavBase)`
  background-color: transparent !important;

  & .nav-link {
    padding: 5px;
  }
`

export const NavItem = styled(NavItemBase)`
  background-color: transparent !important;

  width: 100px;
  text-align: center;

  @media (max-width: 991.98px) {
    margin: 0 auto;
  }

  &.active {
    background-color: ${(props) => props.theme.primary} !important;

    * {
      color: ${(props) => props.theme.white} !important;
    }
  }
`

export const NavLink = styled(NavLinkBase)`
  margin: 0 auto;
  color: ${(props) => props.theme.gray300} !important;
`

export const NavbarToggler = styled(NavbarTogglerBase)`
  border: none;
`

export interface NavEntryProps {
  url: string
  text: string
}

export function NavEntry({ url, text }: NavEntryProps) {
  const { pathname } = useRouter()
  const navItemClassName = useMemo(() => classNames(pathname === url && 'active'), [pathname, url])
  return (
    <NavItem key={url} className={navItemClassName}>
      <NavLink tag={Link} href={url}>
        {text}
      </NavLink>
    </NavItem>
  )
}

export interface NavigationBarProps {
  pathname: string
}

export function NavigationBar() {
  const [isOpen, setIsOpen] = useState(false)
  const toggle = () => setIsOpen(!isOpen)
  const navEntries = useMemo(
    () => Object.entries(navLinksLeft).map(([url, text]) => <NavEntry key={url} url={url} text={text} />),
    [],
  )

  return (
    <Navbar expand="md" color="light" light role="navigation">
      <NavbarToggler onClick={toggle} />

      <Collapse isOpen={isOpen} navbar>
        <Nav navbar>{navEntries}</Nav>
      </Collapse>
    </Navbar>
  )
}
