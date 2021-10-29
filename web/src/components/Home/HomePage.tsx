import React, { useCallback, useEffect, useState } from 'react'

import { Col, Container, Row } from 'reactstrap'

import { ModelResult } from 'src/algorithms/runModel'
import { Uploader } from 'src/components/Common/Uploader'

export async function importAndRunModel() {
  const { runModel } = await import('src/algorithms/runModel')
  return runModel()
}

export function HomePage() {
  const [result, setResult] = useState<ModelResult>()

  const onUpload = useCallback((file: File) => {
    console.warn('Not implemented: onUpload:', { file })
  }, [])

  useEffect(() => {
    // eslint-disable-next-line no-void
    void importAndRunModel()
      .then((result) => setResult(result))
      .catch(console.error)
  }, [])

  return (
    <Container>
      <Row noGutters>
        <Col>
          <Uploader onUpload={onUpload} />
        </Col>
      </Row>

      <Row noGutters>
        <Col>{result && JSON.stringify(result, null, 2)}</Col>
      </Row>
    </Container>
  )
}
