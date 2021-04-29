// Note: Require the cpu and webgl backend and add them to package.json as peer dependencies.
require('@tensorflow/tfjs-node-gpu');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs-extra');
const jpeg = require('jpeg-js');
const Bromise = require('bluebird');
const R = require('ramda');
const sizeOf = require('image-size');
const pathinit = './random_pic/';
const FileName = [];
const object = {
  imageInfo: []
};

const checkifjpg = (x) => x.slice(-4) === '.jpg';

const getfiles = fs.readdirSync(pathinit).forEach((file) => {
  FileName.push(pathinit + file);
});

const createfolder = fs.ensureDir;

const readJpg = async (path) => jpeg.decode(await fs.readFile(path), true);

const movefiles = (nameofile, classname) => {
  fs.rename(
    nameofile,
    `./${classname}/${classname}_${fs.readdirSync(classname).length}.jpg`,
    (error) => {
      if (error) {
        throw error;
      }
    }
  );
};

(async () => {
  getfiles;

  const FileNameFilter = R.filter(checkifjpg, FileName);

  const imgList = await Bromise.map(FileNameFilter, readJpg);

  // Load the model.
  const model = await cocoSsd.load();

  // Classify the image.
  const predictions = await Bromise.map(imgList, (x) => model.detect(x));

  const classname = getclass(predictions);
  const dimension = getbbox(predictions);

  const ziplistdim = R.zip(FileNameFilter, dimension);
  calculdimbbox(ziplistdim);

  const ziplistclass = R.zip(FileNameFilter, classname);
  createandmove(ziplistclass);
})();

const docalculatebbox = ([x, [a, b, c, d]]) => {
  object.imageInfo.push({
    imageName: x,
    originalDim: `${sizeOf(x).width} x ${sizeOf(x).width}`,
    detectionBoxDim: `${Math.round(a + c)} x ${Math.round(b + d)}`,
    ratio:
      (Math.round(a + c) * Math.round(b + d)) /
      (sizeOf(x).width * sizeOf(x).width)
  });
  fs.writeFile(
    'image_information.json',
    JSON.stringify(object, null, '\t'),
    'utf8'
  );
};

const calculdimbbox = R.map(docalculatebbox);

const docreatefolder = (x) =>
  R.pipe(R.nth(1), createfolder, R.andThen(R.always(x)))(x);

const domovefiles = ([x, y]) => movefiles(x, y);

const getclass = R.pipe(R.flatten, R.map(R.prop('class')));

const getbbox = R.pipe(R.flatten, R.map(R.prop('bbox')));

const createandmove = R.map(R.pipe(docreatefolder, R.andThen(domovefiles)));