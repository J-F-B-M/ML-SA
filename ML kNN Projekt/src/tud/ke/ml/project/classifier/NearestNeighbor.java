package tud.ke.ml.project.classifier;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedSet;
import java.util.TreeSet;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set)
 * 
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {

	protected double[] scaling;
	protected double[] translation;

	protected List<List<Object>> traindata;

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> votes;
		if (isInverseWeighting()) {
			votes = getWeightedVotes(subset);
		} else {
			votes = getUnweightedVotes(subset);
		}
		return getWinner(votes);
	}

	@Override
	protected void learnModel(List<List<Object>> traindata) {
		this.traindata = traindata;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> map = new HashMap<Object, Double>();

		for (Pair<List<Object>, Double> pair : subset) {
			Object key = pair.getA().get(getClassAttribute());
			map.put(key, map.get(key) == null ? 1 : map.get(key) + 1);
		}

		return map;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> map = new HashMap<Object, Double>();
		Double result;

		for (Pair<List<Object>, Double> pair : subset) {
			Object key = pair.getA().get(getClassAttribute()); // If any listing exists (which is a sum of inverse values), I add the current inverse value.
			map.put(key, 1 / (pair.getB() + 0.001) + (map.containsKey(key) ? map.get(key) : 0));
		}

		return map;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		Object winner = null;
		double maxValue = Double.MIN_VALUE;

		for (Entry<Object, Double> entry : votesFor.entrySet()) {
			if (entry.getValue() > maxValue) {
				winner = entry.getKey();
				maxValue = entry.getValue();
			}
		}

		return winner;
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		SortedSet<Pair<List<Object>, Double>> sortedSet = new TreeSet<>(new Comparator<Pair<List<Object>, Double>>() {

			@Override
			public int compare(Pair<List<Object>, Double> o1, Pair<List<Object>, Double> o2) {
				if (o1.getB() < o2.getB()) {
					return -1;
				}
				if (o1.getB() == o2.getB()) {
					return 0;
				}
				return 1;
			}
		});

		for (List<Object> trainingData : this.traindata) {
			double distance;
			if (getMetric() == 0) {
				distance = determineManhattanDistance(testdata, trainingData);
			} else {
				distance = determineEuclideanDistance(testdata, trainingData);
			}
			sortedSet.add(new Pair<List<Object>, Double>(trainingData, distance));
		}

		List<Pair<List<Object>, Double>> result = new ArrayList<>(getkNearest());
		Iterator<Pair<List<Object>, Double>> it = sortedSet.iterator();
		for (int i = 0; i < getkNearest()/* && it.hasNext() */; i++) {
			if (!it.hasNext()) {
				System.out.println("Too few samples!");
				break;
			}
			result.add(it.next());
		}
		return result;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		Iterator<Object> it1 = instance1.iterator();
		Iterator<Object> it2 = instance2.iterator();
		double distance = 0;

		while (it1.hasNext() && it2.hasNext()) {
			Object obj1 = it1.next();
			Object obj2 = it2.next();

			if (obj1 instanceof String && obj2 instanceof String) {
				if (!obj1.equals(obj2)) {
					distance += 1;
				}
			} else if (obj1 instanceof Double && obj2 instanceof Double) {
				distance += Math.abs((Double) obj1 - (Double) obj2);
			} else {
				throw new IllegalArgumentException();
			}
		}
		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		Iterator<Object> it1 = instance1.iterator();
		Iterator<Object> it2 = instance2.iterator();
		double distance = 0;

		while (it1.hasNext() && it2.hasNext()) {
			Object obj1 = it1.next();
			Object obj2 = it2.next();

			if (obj1 instanceof String && obj2 instanceof String) {
				if (!obj1.equals(obj2)) {
					distance += 1;
				}
			} else if (obj1 instanceof Double && obj2 instanceof Double) {
				// distance += ((Double) obj1 - (Double) obj2) * ((Double) obj1 - (Double) obj2);
				distance += Math.pow((Double) obj1 - (Double) obj2, 2);
			} else {
				throw new IllegalArgumentException();
			}
		}
		// System.out.println(distance);
		return Math.sqrt(distance);
	}

	@Override
	protected double[][] normalizationScaling() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected String[] getMatrikelNumbers() {
		// FIXME Matrikelnummern eintragen
		return new String[] { "1766932", "1669152", "1664571" };
	}

}
