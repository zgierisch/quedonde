import { httpGet } from './net';
const legacy = require('./legacy');

export class Store {
  constructor(private base: string) {}
}

export function fetchData(path: string) {
  legacy.use(path);
  return httpGet(`${path}`);
}

const helper = () => 42;
