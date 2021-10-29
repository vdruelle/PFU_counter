import React, { PropsWithChildren } from 'react'

import { Col, Container, Row } from 'reactstrap'

import { NavigationBar } from 'src/components/Layout/NavigationBar'

export function Layout({ children }: PropsWithChildren<unknown>) {
  return (
    <Container fluid>
      <Row noGutters>
        <span>
          <NavigationBar />
        </span>
      </Row>

      <Row noGutters>
        <Col>{children}</Col>
      </Row>
    </Container>
  )

  return <>{children}</>
}
