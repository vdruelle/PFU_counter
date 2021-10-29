import React, { useEffect, useState } from 'react'

import { Col, Container, Row } from 'reactstrap'

import { ModelResult } from 'src/algorithms/runModel'

export async function importAndRunModel() {
  const { runModel } = await import('src/algorithms/runModel')
  return runModel()
}

export function HomePage() {
  const [result, setResult] = useState<ModelResult>()

  useEffect(() => {
    // eslint-disable-next-line no-void
    void importAndRunModel()
      .then((result) => setResult(result))
      .catch(console.error)
  }, [])

  return (
    <Container>
      <Row noGutters>
        <Col>{result && JSON.stringify(result, null, 2)}</Col>
      </Row>
    </Container>
  )
}
